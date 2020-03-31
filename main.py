from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    # declare model
    model = PopMusicTransformer(
        is_training=False)
    # generate from scratch
    for i in range(100):
        model.generate(
           n_target_bar=50,
           temperature=1.2,
           output_path='./result/from_scratch' + str(i) + '.midi',
           prompt="evaluation/" + str(i).zfill(3) + ".midi")
 

    model.close()

if __name__ == '__main__':
    main()
