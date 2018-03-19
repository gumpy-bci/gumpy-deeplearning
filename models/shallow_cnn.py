from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, Conv3D


class Shallow_CNN(KerasModel):

    def create_model(self, augmented_data=True, print_summary=False, downsampled=False):

        CLASS_COUNT = 2
        model = Sequential()
        # augmented_data = False
        # print_summary=False

        if augmented_data and downsampled:
            # Conv Block 1
            model.add(Conv2D(input_shape=(3, 512, 1), filters=40, kernel_size=(1, 25), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(3, 488, 40, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=40, kernel_size=(3, 1, 40), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(keras.backend.square))  # custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(488, 40, 1)))
            model.add(Dropout(0.5))
            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(keras.backend.log))  # custom log function
        if augmented_data and not downsampled:
            # Conv Block 1
            model.add(Conv2D(input_shape=(3, 1024, 1), filters=40, kernel_size=(1, 25), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(3, 1000, 40, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=40, kernel_size=(3, 1, 40), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(keras.backend.square))  # custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(1000, 40, 1)))
            model.add(Dropout(0.5))
            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(keras.backend.log))  # custom log function


        else:
            # Conv Block 1
            model.add(Conv2D(input_shape=(3, 1280, 1), filters=40, kernel_size=(1, 25), strides=(1, 1),
                             padding='valid', activation=None))
            model.add(Reshape(target_shape=(3, 1256, 40, 1)))
            model.add(Dropout(0.5))

            # Conv Block 2
            model.add(Conv3D(filters=40, kernel_size=(3, 1, 40), padding='valid',
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(keras.backend.square))  # custom squaring activation function
            model.add(Flatten())
            model.add(Reshape(target_shape=(1256, 40, 1)))
            model.add(Dropout(0.5))

            # Pooling
            model.add(AveragePooling2D(pool_size=(75, 1), strides=(15, 1), data_format='channels_last'))
            model.add(Activation(keras.backend.log))  # custom log function

        # Classification
        model.add(Flatten())
        model.add(Dense(CLASS_COUNT))
        model.add(Activation('softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # assign and return
        self.model = model
        return model

