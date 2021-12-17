import urllib
import warnings
import time
import optuna
import tensorflow
from keras.backend import clear_session
from keras.datasets import cifar10
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, Dropout,MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import logging

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='Test_Optuna_Dec15.txt'
)


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    nb_classes = 10  # dataset dependent

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)
    input_shape = x_train.shape[1:]
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model = Sequential()
    n_layers = trial.suggest_int("num_layers", 1, 5)
    n_activation = trial.suggest_categorical("activation", ["relu","elu", "tanh", "softplus"])
    for i in range(n_layers):
        model.add(
            Conv2D(
                filters=trial.suggest_categorical("n_unit_{}".format(i), [16, 32, 64, 128, 256]),
                kernel_size=(3, 3),
                activation=n_activation,
                padding='same',
                input_shape=input_shape,
                )
        )
        if i < 2:  # otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(trial.suggest_categorical("n_layers", [16, 32, 64, 128, 256]), activation=n_activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation="softmax"))

    # We compile our model with a sampled learning rate.
    model.compile(
        loss="categorical_crossentropy", optimizer=trial.suggest_categorical("optimizer", ["rmsprop", "adam", "sgd", "adagrad"]), metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=trial.suggest_categorical("nb_batch_size", [64, 128]),
        epochs=trial.suggest_categorical("nb_epochs", [64, 128]),
        verbose=1,
    )

    # Evaluate the model accuracy on the validation set.
    score = model.evaluate(x_test, y_test, verbose=1)
    return score[1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

