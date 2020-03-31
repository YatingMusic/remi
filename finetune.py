from model import PopMusicTransformer
from glob import glob
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=True,
        use_chords=True,
        train_from_scratch = False)
    # prepare data
    midi_paths = glob('midi/**/*.midi', recursive=True) + glob('midi/**/*.mid', recursive=True)  # you need to revise it

    if os.path.exists(
            "233.data"):  # Revise the training data save location, as data preprocessing could take hours or even days on commercial hardware.
        with open("233.data", "rb") as file:
            training_data = pickle.load(file)
    else:
        training_data = model.prepare_data(midi_paths=midi_paths)
        with open("233.data", "wb") as file:
            pickle.dump(training_data, file, protocol=4) # For large preprocessed files

    # check output checkpoint folder
    ####################################
    # Restrictions on the folder name is removed.
    ####################################
    output_checkpoint_folder = 'REMI-chord-finetune'  # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)

    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()


if __name__ == '__main__':
    main()
