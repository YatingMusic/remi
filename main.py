from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch.midi',
        prompt=None)
    
    # generate continuation
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5
        output_path='./result/continuation.midi',
        prompt='./data/evaluation/000.midi')
    
    # close model
    model.close()

if __name__ == '__main__':
    main()
