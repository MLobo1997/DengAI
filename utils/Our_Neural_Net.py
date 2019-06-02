from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class Our_Neural_Net(BaseEstimator, RegressorMixin):
    def compile(self):
        self.model = Sequential()
        self.model.add(LSTM(units=self.neurons_layer_0, input_shape=(self.lag, self.nr_of_features)))
        self.model.add(Dense(self.neurons_layer_1))
        if self.nr_layers > 3:
            self.model.add(Dense(self.neurons_layer_1))
        if self.nr_layers > 4:
            self.model.add(Dense(self.neurons_layer_2))
        self.model.add(Dense(self.output_dim))
        self.model.compile(loss=self.loss_fun, optimizer=self.optim_func(self.learn_rate), metrics=['mae', 'mse'])

    def __init__(self, nr_layers=None, neurons_layer_0=None, neurons_layer_1=None, neurons_layer_2=None, learn_rate=0.01, lag=None, nr_of_features=None, optim_func=tf.optimizers.Adam, loss_fun='mse', output_dim=1, epochs=1000, verbose=0, validation_data=None):
        self.nr_layers = nr_layers
        self.neurons_layer_0 = neurons_layer_0
        self.neurons_layer_1 = neurons_layer_1
        self.neurons_layer_2 = neurons_layer_2
        self.learn_rate = learn_rate
        self.lag = lag
        self.nr_of_features = nr_of_features
        self.optim_func = optim_func
        self.loss_fun = loss_fun
        self.output_dim = output_dim
        self.epochs = epochs
        self.verbose = verbose
        self.validation_data = validation_data
        
    def fit(self, X, y):
        self.compile()
        self.history = self.model.fit(X,y, epochs=self.epochs, verbose=self.verbose, shuffle=False, validation_data=self.validation_data)
    
    def predict(self, X):
        return self.model.predict(X)
        