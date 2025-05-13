import numpy as np
import keras
import keras.layers as layers
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score


def read_model(model_file: str = None):
    '''
    This function is for loading an already made model file.

    :param model_file: Optional parameter for the model's file path. If not given in, a model will not be loaded.
    :return: If a model file path was not given, will return nothing. If one was given, will return the loaded model.
    '''
    if model_file is None:
        return None
    return keras.models.load_model(model_file)

# Adapted code from https://stackoverflow.com/a/55435379 for parallel model
class Parallel:
    def __init__(self, model_file: str = None) -> None:
        '''
        Makes the multi-model instance. Loads a model if given a model file path.
        '''
        self.model = read_model(model_file)
        self.history = None

    def train_model(self,
                    train_x_MFCC: np.ndarray,
                    train_x_spectrogram: np.ndarray,
                    train_x_embedding: np.ndarray,
                    train_y: np.ndarray,
                    test_x_MFCC: np.ndarray,
                    test_x_spectrogram: np.ndarray,
                    test_x_embedding: np.ndarray,
                    test_y: np.ndarray,
                    epochs: int = 1,
                    steps_per_epoch: int = 5,
                    batch_size: int = 1,
                    model_file: str = None) -> None:
        
        '''
        Trains a model on the given input data. Will make a model if one is not already loaded.

        :param train_x_MFCC: The training MFCC inputs for the model.
        :param train_x_spectrogram: The training spectrogram inputs for the model.
        :param train_x_embedding: The training YAMNET embedding inputs for the model.
        :param train_y: The training outputs for the model. In this case they should be one-hot encoded labels.
        :param test_x_MFCC: The testing MFCC inputs for the model.
        :param test_x_spectrogram: The testing spectrogram inputs for the model.
        :param test_x_embedding: The testing YAMNET embedding inputs for the model.
        :param test_y: The testing outputs for the model. In this case they should be one-hot encoded labels.
        :param epochs: The number of iterations (epochs) the model should train for.
        :param steps_per_epoch: The number of times the model will train per epoch. Should be the number of training samples divided by the batch size
        :param batch_size: Number of samples to use per epoch step.
        :param model_file: The file path for where the model should be saved.
        :return: Does not return anything, the model gets saved though.
        '''

        if self.model is None:
            # Model 1 is the CNN MFCC model
            model1 = keras.Sequential()
            model1.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2], 1)))
            model1.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
            model1.add(layers.BatchNormalization())
            model1.add(layers.MaxPooling2D())
            model1.add(layers.Dropout(0.3))
            model1.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
            model1.add(layers.BatchNormalization())
            model1.add(layers.MaxPooling2D())
            model1.add(layers.Dropout(0.3))
            model1.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
            model1.add(layers.BatchNormalization())
            model1.add(layers.MaxPooling2D())
            model1.add(layers.Flatten())

            # Model 2 is the CNN spectrogram model
            model2 = keras.Sequential()
            model2.add(layers.Input(shape=(train_x_spectrogram.shape[1], train_x_spectrogram.shape[2], 1)))
            model2.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
            model2.add(layers.BatchNormalization())
            model2.add(layers.MaxPooling2D())
            model2.add(layers.Dropout(0.3))
            model2.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
            model2.add(layers.BatchNormalization())
            model2.add(layers.MaxPooling2D())
            model2.add(layers.Dropout(0.3))
            model2.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
            model2.add(layers.BatchNormalization())
            model2.add(layers.MaxPooling2D())
            model2.add(layers.Flatten())

            # Model 3 is the CNN YAMNET embedding model
            model3 = keras.Sequential()
            model3.add(layers.Input(shape=(train_x_embedding.shape[1], train_x_embedding.shape[2])))
            model3.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding="same"))
            model3.add(layers.BatchNormalization())
            model3.add(layers.MaxPooling1D())
            model3.add(layers.Dropout(0.3))
            model3.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
            model3.add(layers.BatchNormalization())
            model3.add(layers.MaxPooling1D())
            model3.add(layers.Flatten())

            # Model 4 is the LSTM MFCC model
            model4 = keras.Sequential()
            model4.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2])))
            model4.add(layers.Bidirectional(layers.LSTM(units=128, dropout=0.3, kernel_regularizer=l2(0.01), return_sequences=True)))
            model4.add(layers.BatchNormalization())
            model4.add(layers.Bidirectional(layers.LSTM(units=64, dropout=0.3, kernel_regularizer=l2(0.01))))
            model4.add(layers.BatchNormalization())
            model4.add(layers.Flatten())

            # Model 5 is the LSTM spectrogram model
            model5 = keras.Sequential()
            model5.add(layers.Input(shape=(train_x_spectrogram.shape[1], train_x_spectrogram.shape[2])))
            model5.add(layers.Bidirectional(layers.LSTM(units=128, dropout=0.3, kernel_regularizer=l2(0.01), return_sequences=True)))
            model5.add(layers.BatchNormalization())
            model5.add(layers.Bidirectional(layers.LSTM(units=64, dropout=0.3, kernel_regularizer=l2(0.01))))
            model5.add(layers.BatchNormalization())
            model5.add(layers.Flatten())

            # Model 6 is the LSTM YAMNET embedding model
            model6 = keras.Sequential()
            model6.add(layers.Input(shape=(train_x_embedding.shape[1], train_x_embedding.shape[2])))
            model6.add(layers.Bidirectional(layers.LSTM(units=128, dropout=0.3, kernel_regularizer=l2(0.01), return_sequences=True)))
            model6.add(layers.BatchNormalization())
            model6.add(layers.Bidirectional(layers.LSTM(units=64, dropout=0.3, kernel_regularizer=l2(0.01))))
            model6.add(layers.BatchNormalization())
            model6.add(layers.Flatten())

            # Combines the different models and makes 1 prediction
            combined = layers.concatenate(
                [model1.output, model2.output, model3.output, model4.output, model5.output, model6.output])
            x = layers.Dense(128, activation="relu")(combined)
            x = layers.Dense(units=train_y.shape[1], activation="softmax")(x)

            # Makes the overall, parallel model
            model = keras.Model(
                inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input], outputs=x)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.model = model

        # Turned off early stopping so can compare the same number of epochs for different models
       
        # early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        model_checkpoint = ModelCheckpoint(model_file, monitor="val_loss", mode="min", save_best_only=True)

        self.history = self.model.fit(x=[train_x_MFCC, train_x_spectrogram, train_x_embedding,
                                         train_x_MFCC, train_x_spectrogram, train_x_embedding],
                                      y=train_y,
                                      batch_size=batch_size,
                                      validation_data=([test_x_MFCC, test_x_spectrogram, test_x_embedding,
                                                        test_x_MFCC, test_x_spectrogram, test_x_embedding], test_y),
                                      validation_batch_size=batch_size,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=1,
                                      use_multiprocessing=True,
                                      #  callbacks=[early_stopping, model_checkpoint])
                                      callbacks=[model_checkpoint])


    def predict(self, input_MFCC_data: np.ndarray, input_spect_data: np.ndarray, input_embed_data: np.ndarray, true_labels: np.ndarray, batch_size: int = 10) -> np.ndarray:
        '''
        Takes in an array of samples to predict for this model.

        :param input_MFCC_data: Array of sample MFCC inputs.
        :param input_spect_data: Array of sample spectrogram inputs.
        :param input_embed_data: Array of sample YAMNET embedding inputs.
        :param true_labels: True labels for the input data, one-hot encoded.
        :param batch_size: Number of samples to predict at once.
        :return: Just prints out the various stats related to the predictions.
        '''
        predictions =  self.model.predict(x=[input_MFCC_data, input_spect_data, input_embed_data,
                                         input_MFCC_data, input_spect_data, input_embed_data], batch_size=batch_size)

        # Convert predictions and true labels to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(true_labels, axis=1)

        accuracy = accuracy_score(true_classes, predicted_classes)
        class_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        report = classification_report(true_classes, predicted_classes, target_names=class_names)

        print("Classification Report:")
        print(report)
        print(f"Overall Accuracy: {accuracy:.2f}")
