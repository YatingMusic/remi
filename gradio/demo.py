from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import gradio as gr
import requests

url = 'https://github.com/AK391/remi/blob/master/input.midi?raw=true'
r = requests.get(url, allow_redirects=True)
open("input.midi", 'wb').write(r.content)

# declare model
model = PopMusicTransformer(
    checkpoint='REMI-tempo-checkpoint',
    is_training=False)

def inference(midi): 
    # generate continuation
    model.generate(
        n_target_bar=4,
        temperature=1.2,
        topk=5,
        output_path='./result/continuation.midi',
        prompt=midi.name)
    return './result/continuation.midi'
        

title = "Remi"
description = "demo for Remi. To use it, simply upload your midi file, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2002.00212'>Pop Music Transformer: Beat-based Modeling and Generation of Expressive Pop Piano Compositions</a> | <a href='https://github.com/YatingMusic/remi'>Github Repo</a></p>"
examples = [
  ['input.midi']
]
gr.Interface(
    inference, 
    gr.inputs.File(label="Input Midi"), 
    gr.outputs.File(label="Output Midi"),
    title=title,
    description=description,
    article=article,
    examples=examples
    ).launch()