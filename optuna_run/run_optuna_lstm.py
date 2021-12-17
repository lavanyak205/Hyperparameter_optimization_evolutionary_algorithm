import optuna
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from keras.backend import clear_session

from keras.layers import Dense, Flatten, Dropout, LSTM, LeakyReLU
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from numpy import concatenate
from sklearn.metrics import mean_squared_error
scalar = MinMaxScaler()
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')


def create_dataset(dataset, lag=1):
    datax, datay = [], []
    for i in range(len(dataset) - lag - 1):
        a = dataset[i:(i + lag), 0:4]
        datax.append(a)
        datay.append(dataset[i + lag, 0])
    return np.array(datax), np.array(datay)


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    data = pd.read_csv('energydata_complete.csv')

    df_input = data[['Appliances', 'T_out', 'RH_1', 'Visibility']]

    scaled_data = scalar.fit_transform(df_input)

    features = scaled_data
    target = scaled_data[:, 0]
    num_features = 4
    train_size = int(len(features) * 0.8)
    x_train, x_test = features[0:train_size, :], features[train_size:len(features), :]
    y_train, y_test = target[0:train_size], target[train_size:len(features)]

    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_batch_size = trial.suggest_categorical("nb_batch_size", [32, 64])
    n_epochs = trial.suggest_categorical("n_epoch", [10, 15])
    window_length = trial.suggest_categorical("n_window", [1, 3, 6])

    n_optimizer = trial.suggest_categorical("optimizer", ['rmsprop', 'adam', 'sgd', 'adagrad'])

    trainX, trainY = create_dataset(x_train, window_length)
    testX, testY = create_dataset(x_test, window_length)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], num_features))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], num_features))
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(trial.suggest_categorical("n_unit_{}".format(i), [32, 64, 128, 256]),
                           input_shape=(window_length, num_features), return_sequences=True))
        else:
            model.add(LSTM(trial.suggest_categorical("n_unit_{}".format(i), [32, 64, 128, 256]), return_sequences=True))
        model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))
    model.add(LSTM(trial.suggest_categorical("n_units", [32, 64, 128, 256]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=n_epochs, batch_size=n_batch_size, callbacks=[early_stopping], validation_data=(testX, testY),
              verbose=True)
    yhat = model.predict(testX)
    test_X = testX.reshape((testX.shape[0], testX.shape[1] * testX.shape[2]))
    inv_yhat = concatenate((yhat, test_X[:, -3:]), axis=1)
    inv_yhat = scalar.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = testY.reshape((len(testY), 1))
    inv_y = concatenate((test_y, test_X[:, -3:]), axis=1)
    inv_y = scalar.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

    return rmse


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=5000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
