import numpy as np
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import librosa
from librosa.feature import mfcc
import os

from keras.utils import pad_sequences

import keras
import keras.layers as layers
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.LSTM import LSTM
from src.CNN import CNN

DATASET_PATH = "./data/Music"
PICKLE_AUDIO_FILE_PATH = "./data/AudioDataPickle.txt"
PICKLE_MFCC_FILE_PATH = "./data/MFCCPickle.txt"
LSTM_MODEL_PATH = "./models/LSTM.h5"
CNN_MODEL_PATH = "./models/CNN.h5"
NUM_MFCC = 20
RANDOM_STATE = 1000
BATCH_SIZE = 10
EPOCHS = 100


def read_and_save_audio(audio_file_name: str = PICKLE_AUDIO_FILE_PATH) -> None:
    # Just reads in the audio data while also saving the data in a dictionary
    genre_files = dict()
    for genre in os.listdir(DATASET_PATH):
        audio_files = list()
        for audio_file in os.listdir(DATASET_PATH + "/" + genre):
            print(audio_file)
            # open and resample the files from ~22Khz to ~11KHz.
            try:
                audio, sample_rate = librosa.load(DATASET_PATH + "/" + genre + "/" + audio_file)
                audio_files.append(audio)
            except:
                print("File failed to resample: " + audio_file)
        genre_files[genre] = audio_files
    pickle_file = open(audio_file_name, "wb")
    pickle.dump(genre_files, pickle_file)
    pickle_file.close()


def read_saved_audio(file_name: str = PICKLE_AUDIO_FILE_PATH) -> dict:
    # Loads in the saved dictionary of speaker audio data
    file = open(file_name, "rb")
    loaded_audio_data = pickle.load(file)
    file.close()
    return loaded_audio_data


def get_max_audio_length(data: dict) -> int:
    # Need the longest sample for padding others
    max_length = 0
    for genre_data in data.values():
        for sample in genre_data:
            if sample.shape[0] > max_length:
                max_length = sample.shape[0]
    return max_length


def pad_audio(data: dict) -> dict:
    # Need to pad the clips as they are not all exactly the same length
    sample_length = get_max_audio_length(data)
    for genre, genre_data in data.items():
        data[genre] = pad_sequences(genre_data, maxlen=sample_length, dtype='float32', value=0, padding="post")
    return data


def cal_and_save_MFCC(audio: dict, MFCC_file_name: str = PICKLE_MFCC_FILE_PATH) -> None:
    # Calculates the MFCCs of all the audio files
    # Defaults number of MFCCs is 20 so kept that number of them
    genre_MFCC_files = dict()
    for genre, data in audio.items():
        MFCC_files = list()
        for sample in data:
            # Default matrix is num_MFCCs x time_step, want flipped so .T
            mfccs = mfcc(y=sample).T
            MFCC_files.append(mfccs)
        genre_MFCC_files[genre] = MFCC_files

    pickle_file = open(MFCC_file_name, "wb")
    pickle.dump(genre_MFCC_files, pickle_file)
    pickle_file.close()


def read_saved_MFCC(file_name: str = PICKLE_MFCC_FILE_PATH) -> dict:
    # Loads the saved dictionary of saved MFCC values
    file = open(file_name, "rb")
    loaded_MFCC_data = pickle.load(file)
    file.close()
    return loaded_MFCC_data


def convert_data(all_data: dict) -> tuple[np.ndarray, np.ndarray]:
    # Convert the audio data to be in a format that keras model can read easier
    x = []
    y = []
    for (genre, data) in all_data.items():
        for sample in data:
            # don't need the sample rate, thus we only take index 0
            x.append(sample)
            y.append(genre)

    y = np.array(y)
    # Labels are strings for the genres, one-hot encode them numerically
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    x = np.array(x)
    return x, y


def train_and_run_LSTM_model(all_train_x, all_train_y, model_file: str = LSTM_MODEL_PATH,
                             continue_training: bool = False) -> None:
    # Makes, trains, and evaluates the LSTM model

    train_x, test_x, train_y, test_y = train_test_split(all_train_x, all_train_y, train_size=0.8,
                                                        random_state=RANDOM_STATE)
    print(train_y.shape)

    lstm_model = LSTM(model_file if continue_training else None)
    lstm_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE,
                           batch_size=BATCH_SIZE, output_file=model_file, epochs=EPOCHS)
    predictions = lstm_model.evaluate(test_x, test_y)


def train_and_run_CNN_model(all_train_x, all_train_y, model_file: str = CNN_MODEL_PATH,
                            continue_training: bool = False) -> None:
    # Makes, trains, and evaluates the CNN model

    train_x, test_x, train_y, test_y = train_test_split(all_train_x, all_train_y, train_size=0.8,
                                                        random_state=RANDOM_STATE)
    print(train_y.shape)

    cnn_model = CNN(model_file if continue_training else None)
    cnn_model.train_model(train_x, train_y, test_x, test_y, steps_per_epoch=len(train_y) // BATCH_SIZE,
                          batch_size=BATCH_SIZE, output_file=model_file, epochs=EPOCHS)
    predictions = cnn_model.evaluate(test_x, test_y)



if __name__ == '__main__':
    # read_and_save_audio()
    # audio_data = read_saved_audio()
    # audio_data = pad_audio(audio_data)
    # print(audio_data["pop"][99].shape)
    # print(audio_data["blues"][1].shape)
    # cal_and_save_MFCC(audio_data)
    MFCCs = read_saved_MFCC()
    print(MFCCs["pop"][99].shape)
    print(MFCCs["blues"][99].shape)

    all_x, all_y = convert_data(MFCCs)

    train_x, validate_x, train_y, validate_y = train_test_split(all_x, all_y, train_size=0.8, random_state=RANDOM_STATE)

    train_and_run_CNN_model(train_x, train_y)
