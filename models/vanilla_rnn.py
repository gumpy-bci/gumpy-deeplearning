from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense


class Vanilla_RNN(KerasModel):

    def create_model(self, input_shape, num_hidden_neurons=128, num_layers=1, print_summary=False):
        self.model = Sequential()
        if num_layers > 1:
            for i in range(1, num_layers, 1):
                self.model.add(SimpleRNN(num_hidden_neurons, input_shape=input_shape, return_sequences=True))
            self.model.add(SimpleRNN(num_hidden_neurons))
        else:
            self.model.add(SimpleRNN(num_hidden_neurons, input_shape=input_shape))

        self.model.add(Dense(2, activation='softmax'))

        if print_summary:
            print(self.model.summary())

        # compile the model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


        return self.model

