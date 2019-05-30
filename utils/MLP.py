from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_mlp(neurons_per_layer, nr_of_features, optim_func, loss_func):
    network = Sequential()

    #input layer
    network.add(Dense(nr_of_features, activation='sigmoid', input_shape=[nr_of_features]))

    #hidden layers
    for n_neurons in neurons_per_layer:
        network.add(Dense(n_neurons, activation='relu'))

    #output layer
    network.add(Dense(1, activation='relu'))

    network.compile(optimizer=optim_func, loss=loss_func, metrics=['mae'])

    return network