import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils

class PopMusicTransformer(object):
    ########################################
    # initialize
    ########################################
    def __init__(self, checkpoint):
        # load dictionary
        self.dictionary_path = '{}/dictionary.pkl'.format(checkpoint)
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        # model settings
        self.x_len = 512
        self.mem_len = 512
        self.n_layer = 12
        self.d_embed = 512
        self.d_model = 512
        self.n_head = 8
        self.d_head = self.d_model // self.n_head
        self.d_ff = 2048
        self.n_token = len(self.event2word)
        # output settings
        self.batch_size = 1
        # load model
        self.checkpoint_path = '{}/model'.format(checkpoint)
        self.load_model()

    ########################################
    # load model
    ########################################
    def load_model(self):
        # placeholders
        self.x = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.y = tf.compat.v1.placeholder(tf.int32, shape=[self.batch_size, None])
        self.mems_i = [tf.compat.v1.placeholder(tf.float32, [self.mem_len, self.batch_size, self.d_model]) for _ in range(self.n_layer)]
        # model
        initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
        proj_initializer = tf.compat.v1.initializers.random_normal(stddev=0.01, seed=None)
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            xx = tf.transpose(self.x, [1, 0])
            yy = tf.transpose(self.y, [1, 0])
            loss, self.logits, self.new_mem = modules.transformer(
                dec_inp=xx,
                target=yy,
                mems=self.mems_i,
                n_token=self.n_token,
                n_layer=self.n_layer,
                d_model=self.d_model,
                d_embed=self.d_embed,
                n_head=self.n_head,
                d_head=self.d_head,
                d_inner=self.d_ff,
                dropout=0.0,
                dropatt=0.0,
                initializer=initializer,
                proj_initializer=proj_initializer,
                is_training=False,
                mem_len=self.mem_len,
                cutoffs=[],
                div_val=-1,
                tie_projs=[],
                same_length=False,
                clamp_len=-1,
                input_perms=None,
                target_perms=None,
                head_target=None,
                untie_r=False,
                proj_same_dim=True)
        # restore
        self.saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.saver.restore(self.sess, self.checkpoint_path)

    ########################################
    # temperature sampling
    ########################################
    def temperature_sampling(self, logits, temperature, topk=5):
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
        if topk == 1:
            prediction = np.argmax(probs)
        else:
            sorted_index = np.argsort(probs)[::-1]
            candi_index = sorted_index[:topk]
            candi_probs = [probs[i] for i in candi_index]
            # normalize probs
            candi_probs /= sum(candi_probs)
            # choose by predicted probs
            prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
        return prediction

    ########################################
    # extract events for prompt continuation
    ########################################
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        if 'chord' in self.checkpoint_path:
            chord_items = utils.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events

    ########################################
    # generate
    ########################################
    def generate(self, n_target_bar, temperature, output_path, prompt=None):
        # if prompt, load it. Or, random start
        if prompt:
            events = self.extract_events(prompt)
            words = [[self.event2word['{}_{}'.format(e.name, e.value)] for e in events]]
            words[0].append(self.event2word['Bar_None'])
        else:
            words = []
            for _ in range(self.batch_size):
                ws = [self.event2word['Bar_None']]
                if 'chord' in self.checkpoint_path:
                    tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    chords = [v for k, v in self.event2word.items() if 'Chord' in k]
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(chords))
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                else:
                    tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
                    tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
                    ws.append(self.event2word['Position_1/16'])
                    ws.append(np.random.choice(tempo_classes))
                    ws.append(np.random.choice(tempo_values))
                words.append(ws)
        # initialize mem
        batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]
        # generate
        original_length = len(words[0])
        initial_flag = 1
        current_generated_bar = 0
        while current_generated_bar < n_target_bar:
            # input
            if initial_flag:
                temp_x = np.zeros((self.batch_size, original_length))
                for b in range(self.batch_size):
                    for z, t in enumerate(words[b]):
                        temp_x[b][z] = t
                initial_flag = 0
            else:
                temp_x = np.zeros((self.batch_size, 1))
                for b in range(self.batch_size):
                    temp_x[b][0] = words[b][-1]
            # prepare feed dict
            feed_dict = {self.x: temp_x}
            for m, m_np in zip(self.mems_i, batch_m):
                feed_dict[m] = m_np
            # model (prediction)
            _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=feed_dict)
            # sampling
            _logit = _logits[-1, 0]
            word = self.temperature_sampling(
                logits=_logit, 
                temperature=temperature)
            words[0].append(word)
            # if bar event (only work for batch_size=1)
            if word == self.event2word['Bar_None']:
                current_generated_bar += 1
            # re-new mem
            batch_m = _new_mem
        # write
        if prompt:
            utils.write_midi(
                words=words[0][original_length:],
                word2event=self.word2event,
                output_path=output_path,
                prompt_path=prompt)
        else:
            utils.write_midi(
                words=words[0],
                word2event=self.word2event,
                output_path=output_path,
                prompt_path=None)
    
    ########################################
    # close
    ########################################
    def close(self):
        self.sess.close()