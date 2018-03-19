from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, Dropout

class LSTM_STFT(KerasModel):
    def create_model(self, input_shape, num_hidden_neurons=128,
                           num_layers=1, n_dft=128, n_hop=16, dropout=0.0, recurrent_dropout=0.0,
                           print_summary=False):
        model = Sequential()
        # STFT layer
        model.add(Spectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=input_shape,
                              return_decibel_spectrogram=False, power_spectrogram=2.0,
                              trainable_kernel=False, name='static_stft'))

        model.add(Permute((1, 3, 2)))  # needs to be (3,1,2)
        model.add(Reshape((64, 65 * 3)))

        if num_layers > 1:
            for i in range(1, num_layers, 1):
                model.add(LSTM(num_hidden_neurons, return_sequences=True,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout))
            model.add(LSTM(num_hidden_neurons))
        else:
            model.add(LSTM(num_hidden_neurons))

        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # assign and return
        self.model = model
        return model
