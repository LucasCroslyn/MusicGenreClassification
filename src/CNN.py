import numpy as np
import keras
import keras.layers as layers
from keras.callbacks import EarlyStopping, ModelCheckpoint


def read_model(model_file: str = None):
    if model_file is None:
        return None
    return keras.models.load_model(model_file)


class CNN:
    def __init__(self, model_file: str = None) -> None:
        self.model = read_model(model_file)
        self.history = None

    def train_model(self,
                    train_x: np.ndarray,
                    train_y: np.ndarray,
                    test_x: np.ndarray,
                    test_y: np.ndarray,
                    epochs: int = 1,
                    steps_per_epoch: int = 5,
                    batch_size: int = 1,
                    model_file: str = None,
                    feature_type: str = "2d") -> None:

        if self.model is None:
            model = keras.Sequential()
            if feature_type == "2d":
                model.add(layers.Input(shape=(train_x.shape[1], train_x.shape[2], 1)))
                model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling2D())
                model.add(layers.Dropout(0.3))
                model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling2D())
                model.add(layers.Dropout(0.3))
                model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling2D())
            elif feature_type == "1d":
                model.add(layers.Input(shape=(train_x.shape[1], train_x.shape[2])))
                model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding="same"))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D())
                model.add(layers.Dropout(0.3))
                model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
                model.add(layers.BatchNormalization())
                model.add(layers.MaxPooling1D())
            model.add(layers.Flatten())
            model.add(layers.Dense(units=128, activation='relu'))
            model.add(layers.Dense(units=train_y.shape[1], activation="softmax"))
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model.summary()
            self.model = model

        # early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', mode='min', save_best_only=True)

        self.history = self.model.fit(train_x, train_y,
                                 batch_size=batch_size,
                                 validation_data=(test_x, test_y),
                                 validation_batch_size=batch_size,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 verbose=1,
                                 use_multiprocessing=True,
                                #  callbacks=[early_stopping, model_checkpoint])
                                 callbacks=[model_checkpoint])


    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray, batch_size: int = 10) -> float:
        return self.model.evaluate(test_x, test_y, batch_size=batch_size)

    def predict(self, audio_data: np.ndarray, batch_size: int = 10) -> np.ndarray:
        return self.model.predict(audio_data, batch_size=batch_size)
