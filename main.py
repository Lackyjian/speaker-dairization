import os
import shutil


def make_folder(folder_path):
    try:
        os.mkdir(folder_path)
        print(folder_path, "was created ...")
    except:
        print("Exception raised: ", folder_path, "could not be created ...")


def manage_data(data_folder_name):
    # make training and testing folders
    make_folder("TrainingData")
    make_folder("TestingData")
    make_folder("TrainingData/females")
    make_folder("TrainingData/males")
    make_folder("TestingData/females")
    make_folder("TestingData/males")

    training_audios = os.listdir('SLR45/Train/')
    testing_audios = os.listdir('SLR45/Test/')

    for audio in training_audios:
        if '(F)' in audio:
            shutil.copy('SLR45/Train/' + audio, 'TrainingData/females/' + audio)
        if '(M)' in audio:
            shutil.copy('SLR45/Train/' + audio, 'TrainingData/males/' + audio)

    for audio in testing_audios:
        if '(F)' in audio:
            shutil.copy('SLR45/Test/' + audio, 'TestingData/females/' + audio)
        if '(M)' in audio:
            shutil.copy('SLR45/Test/' + audio, 'TestingData/males/' + audio)


if __name__ == "__main__":
    manage_data('SLR45')
