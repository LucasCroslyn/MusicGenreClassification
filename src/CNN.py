import numpy as np
import keras
import keras.layers as layers
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


class CNN:
    def __init__(self, model_file: str = None) -> None:
        '''
        Makes the CNN instance. Loads a model if given a model file path.
        '''
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
        
        '''
        Trains a model on the given input data. Will make a model if one is not already loaded.

        :param train_x: The training inputs for the model.
        :param train_y: The training outputs for the model. In this case they should be one-hot encoded labels.
        :param test_x: The testing inputs for the model.
        :param test_y: The testing outputs for the model. In this case they should be one-hot encoded labels.
        :param epochs: The number of iterations (epochs) the model should train for.
        :param steps_per_epoch: The number of times the model will train per epoch. Should be the number of training samples divided by the batch size
        :param batch_size: Number of samples to use per epoch step.
        :param model_file: The file path for where the model should be saved.
        :param feature_type: The model that will be made depends on the type of input feature being fed in (the YAMNET embeddings are more 1D, instead of 2D).
        :return: Does not return anything, the model gets saved though.
        '''

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

        # Turned off early stopping so can compare the same number of epochs for different models

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
        

    def predict(self, input_data: np.ndarray, true_labels: np.ndarray, batch_size: int = 10) -> np.ndarray:
        '''
        Takes in an array of samples to predict for this model and prints evaluation metrics.

        :param input_data: Array of sample inputs in which label predictions will be done.
        :param true_labels: True labels for the input data, one-hot encoded.
        :param batch_size: Number of samples to predict at once.
        :return: Just prints out the various stats related to the predictions.
        '''
        predictions = self.model.predict(input_data, batch_size=batch_size)
        
        # Convert predictions and true labels to class indices
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(true_labels, axis=1)

        accuracy = accuracy_score(true_classes, predicted_classes)
        class_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        report = classification_report(true_classes, predicted_classes, target_names=class_names)

        print("Classification Report:")
        print(report)
        print(f"Overall Accuracy: {accuracy:.2f}")