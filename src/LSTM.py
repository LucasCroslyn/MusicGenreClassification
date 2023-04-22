import keras
import keras.layers as layers
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_model(model_file: str = None):
    if model_file is None:
        return None
    return keras.models.load_model(model_file)


class LSTM:
    def __init__(self, model_file: str = None) -> None:
        self.model = read_model(model_file)

    def train_model(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    epochs: int = 1,
                    steps_per_epoch: int = 5,
                    batch_size: int = 1,
                    model_file: str = None) -> None:

        if self.model is None:
            model = keras.Sequential()
            model.add(layers.Input(shape=(train_x.shape[1], train_x.shape[2])))
            model.add(layers.LSTM(units=128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model.add(layers.BatchNormalization())
            model.add(layers.LSTM(units=64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
            model.add(layers.BatchNormalization())
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation="relu"))
            model.add(layers.Dense(units=train_y.shape[1], activation="softmax"))
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            self.model = model

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        if model_file is not None:
            model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', save_best_only=True)

        self.model.fit(train_x, train_y,
                       batch_size=batch_size,
                       validation_data=(test_x, test_y),
                       validation_batch_size=batch_size,
                       epochs=epochs,
                       steps_per_epoch=steps_per_epoch,
                       verbose=1,
                       use_multiprocessing=True,
                       callbacks=[early_stopping, model_checkpoint])

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray, batch_size: int = 10) -> float:
        return self.model.evaluate(test_x, test_y, batch_size=batch_size)

    def predict(self, audio_data: np.ndarray, batch_size: int = 10) -> np.ndarray:
        return self.model.predict(audio_data, batch_size=batch_size)
