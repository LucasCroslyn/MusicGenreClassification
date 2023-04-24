import keras
import keras.layers as layers
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_model(model_file: str = None):
    if model_file is None:
        return None
    return keras.models.load_model(model_file)

# Adapted code from https://stackoverflow.com/a/55435379 for parallel model
class Parallel:
    def __init__(self, model_file: str = None) -> None:
        self.model = read_model(model_file)
        self.history = None

    def train_model(self,
                    train_x_MFCC: np.ndarray,
                    train_x_embedding: np.ndarray,
                    train_x_spectrogram: np.ndarray,
                    train_y: np.ndarray,
                    test_x_MFCC: np.ndarray,
                    test_x_embedding: np.ndarray,
                    test_x_spectrogram: np.ndarray,
                    test_y: np.ndarray,
                    epochs: int = 1,
                    steps_per_epoch: int = 5,
                    batch_size: int = 1,
                    model_file: str = None) -> None:

        if self.model is None:
            model1 = keras.Sequential()
            model1.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2], 1)))
            model1.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="valid"))
            model1.add(layers.BatchNormalization())
            model1.add(layers.MaxPooling2D())
            model1.add(layers.Dropout(0.3))
            model1.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
            model1.add(layers.BatchNormalization())
            model1.add(layers.MaxPooling2D())
            model1.add(layers.Dropout(0.3))
            model1.add(layers.Flatten())

            model2 = keras.Sequential()
            model2.add(layers.Input(shape=(train_x_embedding.shape[1], train_x_embedding.shape[2], 1)))
            model2.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="valid"))
            model2.add(layers.BatchNormalization())
            model2.add(layers.MaxPooling2D())
            model2.add(layers.Dropout(0.3))
            model2.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
            model2.add(layers.BatchNormalization())
            model2.add(layers.MaxPooling2D())
            model2.add(layers.Dropout(0.3))
            model2.add(layers.Flatten())

            model3 = keras.Sequential()
            model3.add(layers.Input(shape=(train_x_spectrogram.shape[1], train_x_spectrogram.shape[2], 1)))
            model3.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="valid"))
            model3.add(layers.BatchNormalization())
            model3.add(layers.MaxPooling2D())
            model3.add(layers.Dropout(0.3))
            model3.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
            model3.add(layers.BatchNormalization())
            model3.add(layers.MaxPooling2D())
            model3.add(layers.Dropout(0.3))
            model3.add(layers.Flatten())

            model4 = keras.Sequential()
            model4.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2])))
            model4.add(layers.LSTM(units=128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model4.add(layers.BatchNormalization())
            model4.add(layers.LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model4.add(layers.BatchNormalization())
            model4.add(layers.Flatten())

            model5 = keras.Sequential()
            model5.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2])))
            model5.add(layers.LSTM(units=128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model5.add(layers.BatchNormalization())
            model5.add(layers.LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model5.add(layers.BatchNormalization())
            model5.add(layers.Flatten())

            model6 = keras.Sequential()
            model6.add(layers.Input(shape=(train_x_MFCC.shape[1], train_x_MFCC.shape[2])))
            model6.add(layers.LSTM(units=128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model6.add(layers.BatchNormalization())
            model6.add(layers.LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model6.add(layers.BatchNormalization())
            model6.add(layers.Flatten())

            combined = layers.concatenate(
                [model1.output, model2.output, model3.output, model4.output, model5.output, model6.output])
            x = layers.Dense(128, activation="relu")(combined)
            x = layers.Dense(units=train_y.shape[1], activation="softmax")(x)

            model = keras.Model(
                inputs=[model1.input, model2.input, model3.input, model4.input, model5.input, model6.input])
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.model = model

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        if model_file is not None:
            model_checkpoint = ModelCheckpoint(model_file, monitor="val_loss", mode="min", save_best_only=True)

        self.history = self.model.fit(x=[train_x_MFCC, train_x_embedding, train_x_spectrogram,
                                         train_x_MFCC, train_x_embedding, train_x_spectrogram],
                                      y=train_y,
                                      batch_size=batch_size,
                                      validation_data=([test_x_MFCC, test_x_embedding, test_x_spectrogram,
                                                        test_x_MFCC, test_x_embedding, test_x_spectrogram], test_y),
                                      validation_batch_size=batch_size,
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      verbose=1,
                                      use_multiprocessing=True,
                                      callbacks=[early_stopping, model_checkpoint])

    def evaluate(self, x, y: np.ndarray, batch_size: int = 10) -> float:
        return self.model.evaluate(x=x, y=y, batch_size=batch_size)

    def predict(self, audio_data: np.ndarray, batch_size: int = 10) -> np.ndarray:
        return self.model.predict(audio_data, batch_size=batch_size)
