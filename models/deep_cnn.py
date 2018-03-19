from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, Conv3D


class Deep_CNN(KerasModel):

    def create_model(self, augmented_data=False, print_summary=False, downsampled=False):
        CLASS_COUNT = 2

        model = Sequential()
        if augmented_data and downsampled:
            input_shape = (3, 1280, 1)
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))
            model.add(Reshape(target_shape=(3, 1271, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1271, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(138, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(43, 100, 1)))
            model.add(Dropout(0.5))

        if augmented_data and not downsampled:
            input_shape = (3, 1024, 1)
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))

            model.add(Reshape(target_shape=(3, 1015, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1015, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            model.add(Reshape(target_shape=(109, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(33, 100, 1)))
            model.add(Dropout(0.5))

        if not augmented_data and not downsampled:
            input_shape = (3, 2560, 1)

            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))

            model.add(Reshape(target_shape=(3, 2551, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(2551, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            model.add(Reshape(target_shape=(280, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())

            model.add(Reshape(target_shape=(90, 100, 1)))
            model.add(Dropout(0.5))

        if not augmented_data and downsampled:
            # Conv Pool Block 1
            model.add(Conv2D(input_shape=input_shape, filters=25, kernel_size=(1, 10), strides=(1, 1),
                             padding='valid', activation='linear'))
            model.add(Reshape(target_shape=(3, 1271, 25, 1)))
            model.add(Conv3D(filters=25, kernel_size=(3, 1, 25),
                             data_format='channels_last'))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(Flatten())
            model.add(Reshape(target_shape=(1271, 25, 1)))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1), data_format='channels_last'))
            model.add(Dropout(0.5))

            # Conv Pool Block 2
            model.add(Conv2D(filters=50, kernel_size=(10, 25)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(138, 50, 1)))
            model.add(Dropout(0.5))

            # Conv Pool Block 3
            model.add(Conv2D(filters=100, kernel_size=(10, 50)))
            model.add(BatchNormalization())
            model.add(Activation(activation='elu'))
            model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))
            model.add(Flatten())
            model.add(Reshape(target_shape=(43, 100, 1)))
            model.add(Dropout(0.5))

        # Conv Pool Block 4
        model.add(Conv2D(filters=200, kernel_size=(10, 100)))

        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(MaxPool2D(pool_size=(3, 1), strides=(3, 1)))

        # Softmax for classification
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
