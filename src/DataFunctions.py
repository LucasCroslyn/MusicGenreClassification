import os
import pickle
import librosa
import numpy as np
import tensorflow_hub as hub
from sklearn.preprocessing import LabelBinarizer
from keras.utils import pad_sequences
from collections import defaultdict
from typing import Any

def save_pickle(data: Any, pickle_path: str) -> None:
	'''
	Saves the data into a pickled file for compacted storage.

	:param data: The data that should be pickled and saved.
	:param pickle_path: The file path for where the pickled data should be saved to.
	:return: Does not return the data that was pickled, just saved it.
	'''
	pickle_file = open(pickle_path, "wb")
	pickle.dump(data, pickle_file)
	pickle_file.close()


def read_pickle(file_path: str) -> Any:
	'''
	Loads in the data from a saved pickle file.

	:param file_path: The path to the pickle file.
	:return data: Returns the data that was contained within the pickle file. 
	'''
	file = open(file_path, "rb")
	data = pickle.load(file)
	file.close()
	return data


def read_save_audio(dataset_folder_path: str, pickle_path: str, sample_rate: int = 16000, duration: float = 30.0) -> None:
	'''
	Reads in the audio files (converts to mono channel, not multi-channel) and saves them into a pickled dictionary

	:param dataset_folder_path: The path to the main folder that contains the subfolders.
	:param pickle_path: The location to save the pickled dataset in.
	:param sample_rate: The sampling rate to load the audio in. Default is 16KHz.
	:param duration: The length in seconds of how much audio to load for each sample.
	:return: Does not return anything, saves the data file instead.
	'''
	genre_audio_dict = defaultdict(list)
	target_length = int(sample_rate * duration)
	for genre in os.listdir(dataset_folder_path):
		for audio_file in os.listdir(dataset_folder_path + "/" + genre):
			print(audio_file)
			try:
				audio, _ = librosa.load(dataset_folder_path + "/" + genre + "/" + audio_file, sr=sample_rate, duration=duration)
				# If the audio file is too short, pad it with silence
				if len(audio) < target_length:
					audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
				
				genre_audio_dict[genre].append(audio)
			
			except Exception as e:
				print(f"{audio_file} failed to load: {e}")
	
	save_pickle(data=genre_audio_dict, pickle_path=pickle_path)


def cal_save_MFCC(dataset: dict[list[np.ndarray]], pickle_path: str, num_MFCC: int=20, sample_rate: int=16000) -> None:
	'''
	Calculates the Mel-Frequency Cepstral Coefficients (MFCCs) features of the audio samples.
	This data is a way to reduce the space of the data while transforming the data into more relevant information.
	The data gets saved as a dictionary as a pickle file.

	:param dataset: The dataset of the raw (but re-sampled) audio.
	:param pickle_path: The path to where the MFCC data should be stored as a pickled file.
	:param num_MFCC: The number of MFCCs that should be calculated per sample.
	:param sample_rate: The sample rate that the audio was converted to.
	:return: Does not return the MFCC data but saves it instead.
	'''

	genre_MFCC_dict = defaultdict(list)

	for genre, samples in dataset.items():
		print(genre)
		for sample in samples:
			genre_MFCC_dict[genre].append(librosa.feature.mfcc(y=sample, sr=sample_rate, n_mfcc=num_MFCC).T)
	
	save_pickle(data=genre_MFCC_dict, pickle_path=pickle_path)


def cal_save_embedding_spectrogram(dataset: dict[list[np.ndarray]], pickle_embed_path: str, pickle_spect_path: str) -> None:
	'''
	Uses the YAMNET model to obtain embeddings and the spectrogram for each audio sample.
	The embeddings and spectrograms get saved into their own dictionaries as pickle files.

	:param dataset: The dataset of the raw (but re-sampled) audio.
	:param pickle_embed_path: The path to where the embedding data should be stored as a pickled file.
	:param pickle_spect_path: The path to where the spectrogram data should be stored as a pickled file.
	:return: Does not return the data but saves it instead.
	'''

	genre_embedding_dict = defaultdict(list)
	genre_spectrogram_dict = defaultdict(list)

	yamnet_model_path = "https://tfhub.dev/google/yamnet/1"
	yamnet_model = hub.load(yamnet_model_path)

	for genre, samples in dataset.items():
		print(genre)
		for sample in samples:
			_, embedding, spectrogram = yamnet_model(sample)
			genre_embedding_dict[genre].append(np.array(embedding))
			genre_spectrogram_dict[genre].append(np.array(spectrogram))

	save_pickle(data=genre_embedding_dict, pickle_path=pickle_embed_path)
	save_pickle(data=genre_spectrogram_dict, pickle_path=pickle_spect_path)


def split_dataset(dataset: dict[list[np.ndarray]], num_sections: int) -> dict[list[np.ndarray]]:
	'''
	Splits each sample into multiple, smaller, sub-samples.

	:param dataset: The dataset to which each sample will be split.
	:param num_sections: The number of sub-samples to make per sample.
	:return: Returns the new dataset with more, but smaller, samples.
	'''
	new_dataset = {}
	for genre, samples in dataset.items():
		new_samples = []
		for sample in samples:
			new_samples.extend(np.array_split(sample, indices_or_sections=num_sections))
		new_dataset[genre] = new_samples
		print(len(new_samples))
	return new_dataset


def convert_data(dataset: dict[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
	'''
	Separate the dataset dictionary into an array for the inputs and an array for outputs (genre labels).
	Paddings will be done as well if samples don't have the same shape (only actually matters for MFCCs).
	The labels are one-hot encoded as well.

	:param dataset: The dataset to convert.
	:return: Returns both the input array and the output array in a tuple.
	'''
	x = []
	y = []

	for genre, samples in dataset.items():
		for sample in samples:
			y.append(genre)
			x.append(sample)
	
	x = np.array(x, dtype="float32")
	y = np.array(y)
	y = LabelBinarizer().fit_transform(y)
	return x, y