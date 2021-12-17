import tensorflow
import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM, LeakyReLU, Dropout, Dense
from sklearn.metrics import mean_squared_error
from keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam
import genome
from numpy import concatenate

import logging

scalar = MinMaxScaler()
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min')


def create_dataset(dataset, lag=1):
    datax, datay = [], []
    for i in range(len(dataset) - lag - 1):
        a = dataset[i:(i + lag), 0:4]
        datax.append(a)
        datay.append(dataset[i + lag, 0])
    return np.array(datax), np.array(datay)


def get_lstm_params():
    data = pd.read_csv('energydata_complete.csv')

    df_input = data[['Appliances', 'T_out', 'RH_1', 'Visibility']]

    scaled_data = scalar.fit_transform(df_input)

    features = scaled_data
    target = scaled_data[:, 0]
    num_features = 4
    train_size = int(len(features) * 0.8)
    x_train, x_test = features[0:train_size, :], features[train_size:len(features), :]
    y_train, y_test = target[0:train_size], target[train_size:len(features)]
    return (x_train, x_test, y_train, y_test, num_features)


def get_lstm_model(genomes, window_length, num_features):
    n_layers = genomes.geneparam['nb_layers']
    nb_neurons = genomes.nb_neurons()
    n_optimizer = genomes.geneparam['optimizer']
    logging.info("Neurons and layer:%s,%d" % (str(nb_neurons), n_layers))

    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            model.add(LSTM(nb_neurons[i], input_shape=(window_length, num_features), return_sequences=True))
        else:
            model.add(LSTM(nb_neurons[i], return_sequences=True))
        model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.2))
    model.add(LSTM(nb_neurons[len(nb_neurons) - 1], return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=n_optimizer)
    return model


def train_lstm_model(genomes):
    n_batch_size = genomes.geneparam['nb_batch_size']
    n_epochs = genomes.geneparam['n_epoch']
    n_window_length = genomes.geneparam['n_window_size']
    logging.info("Batch size, epoch, window_length:%s,%d %d" % (str(n_batch_size), n_epochs, n_window_length))

    x_train, x_test, y_train, y_test, num_features = get_lstm_params()

    trainX, trainY = create_dataset(x_train, n_window_length)
    testX, testY = create_dataset(x_test, n_window_length)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], num_features))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], num_features))

    model = get_lstm_model(genomes, n_window_length, num_features)
    model.fit(trainX, trainY, epochs=1, batch_size=1, callbacks=[early_stopping], validation_data=(testX, testY),
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
    logging.info("Test RMSE: %.3f" % rmse)
    return rmse
