import tensorflow as tf

def embedding_lookup(lookup_table, x):
    return tf.compat.v1.nn.embedding_lookup(lookup_table, x)


def normal_embedding_lookup(x, n_token, d_embed, d_proj, initializer,
                            proj_initializer, scope='normal_embed', **kwargs):
    emb_scale = d_proj ** 0.5
    with tf.compat.v1.variable_scope(scope):
        lookup_table = tf.compat.v1.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
        y = embedding_lookup(lookup_table, x)
        if d_proj != d_embed:
            proj_W = tf.compat.v1.get_variable('proj_W', [d_embed, d_proj], initializer=proj_initializer)
            y = tf.einsum('ibe,ed->ibd', y, proj_W)
        else:
            proj_W = None
        ret_params = [lookup_table, proj_W]
    y *= emb_scale
    return y, ret_params


def normal_softmax(hidden, target, n_token, params, scope='normal_softmax', **kwargs):
    def _logit(x, W, b, proj):
        y = x
        if proj is not None:
            y = tf.einsum('ibd,ed->ibe', y, proj)
        return tf.einsum('ibd,nd->ibn', y, W) + b

    params_W, params_projs = params[0], params[1]

    with tf.compat.v1.variable_scope(scope):
        softmax_b = tf.compat.v1.get_variable('bias', [n_token], initializer=tf.zeros_initializer())
        output = _logit(hidden, params_W, softmax_b, params_projs)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
    return nll, output


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]


def positionwise_FF(inp, d_model, d_inner, dropout, kernel_initializer,
                    scope='ff', is_training=True):
    output = inp
    with tf.compat.v1.variable_scope(scope):
        output = tf.keras.layers.Dense(d_inner, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_1')(inp)
        output = tf.keras.layers.Dropout(dropout, name='drop_1')(output, training=is_training)
        output = tf.keras.layers.Dense(d_model, activation=tf.nn.relu, 
                                       kernel_initializer=kernel_initializer, name='layer_2')(output)
        output = tf.keras.layers.Dropout(dropout, name='drop_2')(output, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(output + inp)
    return output


def _create_mask(qlen, mlen, same_length=False):
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret


def _cache_mem(curr_out, prev_mem, mem_len=None):
    if mem_len is None or prev_mem is None:
        new_mem = curr_out
    elif mem_len == 0:
        return prev_mem
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
    return tf.stop_gradient(new_mem)


def rel_shift(x):
    x_size = tf.shape(x)
    x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
    x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, x_size)
    return x


def rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model,
                       n_head, d_head, dropout, dropatt, is_training,
                       kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.compat.v1.variable_scope(scope):
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w], 0) if mems is not None and mems.shape.ndims > 1 else w

        w_heads = tf.keras.layers.Dense(3 * n_head * d_head, use_bias=False, 
                                        kernel_initializer=kernel_initializer, name='qkv')(cat)
        r_head_k = tf.keras.layers.Dense(n_head * d_head, use_bias=False,
                                         kernel_initializer=kernel_initializer, name='r')(r)
        
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        r_head_k = tf.reshape(r_head_k, [rlen, n_head, d_head])

        rw_head_q = w_head_q + r_w_bias
        rr_head_q = w_head_q + r_r_bias

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = tf.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = rel_shift(BD)

        attn_score = (AC + BD) * scale
        attn_mask_t = attn_mask[:, :, None, None]
        attn_score = attn_score * (1 - attn_mask_t) - 1e30 * attn_mask_t

        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.keras.layers.Dropout(dropatt)(attn_prob, training=is_training)

        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.keras.layers.Dense(d_model, use_bias=False, 
                                         kernel_initializer=kernel_initializer, name='o')(attn_vec)
        attn_out = tf.keras.layers.Dropout(dropout)(attn_out, training=is_training)
        output = tf.keras.layers.LayerNormalization(axis=-1)(attn_out + w)
        return output


def transformer(dec_inp, target, mems, n_token, n_layer, d_model, d_embed,
                n_head, d_head, d_inner, dropout, dropatt,
                initializer, is_training, proj_initializer=None,
                mem_len=None, cutoffs=[], div_val=1, tie_projs=[],
                same_length=False, clamp_len=-1,
                input_perms=None, target_perms=None, head_target=None,
                untie_r=False, proj_same_dim=True,
                scope='transformer'):
    """
    cutoffs: a list of python int. Cutoffs for adaptive softmax.
    tie_projs: a list of python bools. Whether to tie the projections.
    perms: a list of tensors. Each tensor should of size [len, bsz, bin_size].
        Only used in the adaptive setting.
    """
    new_mems = []
    with tf.compat.v1.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_layer, n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_layer, n_head, d_head], initializer=initializer)
        else:
            r_w_bias = tf.compat.v1.get_variable('r_w_bias', [n_head, d_head], initializer=initializer)
            r_r_bias = tf.compat.v1.get_variable('r_r_bias', [n_head, d_head], initializer=initializer)

        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = qlen + mlen

        if proj_initializer is None:
            proj_initializer = initializer

        embeddings, shared_params = normal_embedding_lookup(
            x=dec_inp,
            n_token=n_token,
            d_embed=d_embed,
            d_proj=d_model,
            initializer=initializer,
            proj_initializer=proj_initializer)
        
        attn_mask = _create_mask(qlen, mlen, same_length)
        
        pos_seq = tf.range(klen - 1, -1, -1.0)
        if clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
        pos_emb = positional_embedding(pos_seq, inv_freq)

        output = tf.keras.layers.Dropout(rate=dropout)(embeddings, training=is_training)
        pos_emb = tf.keras.layers.Dropout(rate=dropout)(pos_emb, training=is_training)

        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output, mems[i], mem_len))

            with tf.compat.v1.variable_scope('layer_{}'.format(i)):
                output = rel_multihead_attn(
                    w=output,
                    r=pos_emb,
                    r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                    r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                    attn_mask=attn_mask,
                    mems=mems[i],
                    d_model=d_model,
                    n_head=n_head,
                    d_head=d_head,
                    dropout=dropout,
                    dropatt=dropatt,
                    is_training=is_training,
                    kernel_initializer=initializer)

                output = positionwise_FF(
                    inp=output,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    is_training=is_training)

        output = tf.keras.layers.Dropout(dropout)(output, training=is_training)

        loss, logits = normal_softmax(
            hidden=output,
            target=target,
            n_token=n_token,
            params=shared_params)

        return loss, logits, new_mems