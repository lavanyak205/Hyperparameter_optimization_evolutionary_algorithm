
import random
import numpy as np
import tensorflow
import logging
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from pso import Pso

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='particle_swarm.txt'
)

num_classes = 10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices

all_possible_genes = {
    'nb_neurons': [16, 32, 64, 128, 256],
    'nb_layers': [1, 3, 5],
    'nb_batch_size': [128, 256],
    'n_epoch': [128, 256],
    'activation': ['relu', 'elu', 'tanh', 'softplus'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad']
    # 'optimizer': ['adam']
}
hyper_parameters = {
    'nb_neurons':[1,5],
    'nb_layers':[1,5],
    'nb_batch_size':[1,2],
    'n_epoch':[1,2],
    'activation': ['relu', 'elu', 'tanh', 'softplus'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad']
}
n_neurons ={1:16, 2:32, 3:64, 4:128, 5:256}
n_batch_size = {1:128, 2:256}
n_epoch = {1:4,2:5}
n_activation = {1:'relu', 2:'elu', 3:'tanh', 4:'softplus'}
n_optimizer = {1: 'rmsprop', 2: 'adam', 3: 'sgd', 4: 'adagrad'}
def func(x):

    n_list = []
    for i in range(len(n_neurons)):
        n_list.append(x[i])
    t = len(n_list)
    l, b, e, r, o = x[len(n_list)],x[len(n_list)+1],x[len(n_list)+2],x[len(n_list)+3],x[len(n_list)+4]


    model = Sequential()
    t = True
    # Add each layer.
    for i in range(0, l):
        if i == 0:
            model.add(Conv2D(n_neurons[n_list[i]], kernel_size=(3, 3), activation=n_activation[r], padding='same',input_shape=input_shape))
        else:
            model.add(Conv2D(n_neurons[n_list[i]], kernel_size=(3, 3), activation=n_activation[r], padding='same'))

        if i < 2:  # otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # always use last nb_neurons value for dense layer
    #  model.add(Dense(nb_neurons[len(nb_neurons) - 1], activation=activation))

    model.add(Dense(n_neurons[n_list[-1]], activation=n_activation[r]))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=n_optimizer[o], metrics=['accuracy'])

    cp = [tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')]

    model.fit(x_train, y_train,
              batch_size=n_batch_size[b],
              epochs=n_epoch[e],
              verbose=1,
              validation_data=(x_test, y_test), callbacks=cp)

    score = model.evaluate(x_test, y_test, verbose=1)

    # loss, val
    print('current config:', x, 'val:', score[1])
    return score[1]


if __name__ == "__main__":
    pso = Pso(swarmsize=5, maxiter=2)
# n,sf,sp,l
    bp, value = pso.run(func, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5,5,5,5,5, 3, 2, 2, 4,4])

    v = func(bp)

    print('Test loss:', bp)
    print('Test accuracy:', value, v)