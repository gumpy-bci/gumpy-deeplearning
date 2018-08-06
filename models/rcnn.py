from .model import KerasModel
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import merge, Conv2D, MaxPooling2D, Input
from keras.layers.advanced_activations import PReLU
from keras.layers import Activation
from keras.models import Model


class RCNN(KerasModel):

    # TODO: why is this called RCL?
    def RCL(self,l, a):
        # TODO: documentation

        # first convolutional layer
        conv1 = Conv2D(filters=128, kernel_size=(1, 9), strides=(1, 1), padding='same', data_format='channels_last',
                       init='he_normal')(l)
        bn1 = BatchNormalization(epsilon=0.000001)(conv1)
        relu1 = PReLU()(bn1)
        pool1 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid', data_format='channels_last')(relu1)
        drop1 = Dropout(0)(pool1)

        # start first RCL layer
        # the second time convolution and stored for recurrent
        conv2 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', init='he_normal')(drop1)
        bn2 = BatchNormalization(axis=1, epsilon=0.000001)(conv2)
        relu2 = PReLU()(bn2)

        # first recurrent for the first convolution
        conv2a = Conv2D(filters=128, kernel_size=(1, 9), padding='same', init='he_normal')
        conv2aa = conv2a(relu2)
        merged2a = merge([conv2, conv2aa], mode='sum')

        # second recurrent for the first convolution
        bn2a = BatchNormalization(axis=1, epsilon=0.000001)(merged2a)
        relu2a = PReLU()(bn2a)
        conv2b = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv2a.get_weights())(relu2a)
        merged2b = merge([conv2, conv2b], mode='sum')

        # third recurrent for the first convolution
        bn2b = BatchNormalization(axis=1, epsilon=0.000001)(merged2b)
        relu2b = PReLU()(bn2b)
        conv2c = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv2a.get_weights())(relu2b)
        merged2c = merge([conv2, conv2c], mode='sum')

        bn2c = BatchNormalization(axis=1, epsilon=0.000001)(merged2c)
        relu2c = PReLU()(bn2c)
        pool2 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid', data_format='channels_last')(relu2c)
        drop2 = Dropout(0.2)(pool2)

        conv3 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(drop2)
        bn3 = BatchNormalization(axis=1, epsilon=0.000001)(conv3)
        relu3 = PReLU()(bn3)
        conv3a = Conv2D(filters=128, kernel_size=(1, 9), padding='same', init='he_normal')
        conv3aa = conv3a(relu3)
        merged3a = merge([conv3, conv3aa], mode='sum')

        bn3a = BatchNormalization(axis=1, epsilon=0.000001)(merged3a)
        relu3a = PReLU()(bn3a)
        conv3b = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv3a.get_weights())(relu3a)
        merged3b = merge([conv3, conv3b], mode='sum')

        bn3b = BatchNormalization(axis=1, epsilon=0.000001)(merged3b)
        relu3b = PReLU()(bn3b)
        conv3c = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv3a.get_weights())(relu3b)
        merged3c = merge([conv3, conv3c], mode='sum')

        bn3c = BatchNormalization(axis=1, epsilon=0.000001)(merged3c)
        relu3c = PReLU()(bn3c)
        pool3 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid', data_format='channels_last')(relu3c)
        drop3 = Dropout(0.2)(pool3)

        conv4 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', init='he_normal')(drop3)
        bn4 = BatchNormalization(axis=1, epsilon=0.000001)(conv4)
        relu4 = PReLU()(bn4)
        conv4a = Conv2D(filters=128, kernel_size=(1, 9), padding='same')
        conv4aa = conv4a(relu4)
        merged4a = merge([conv4, conv4aa], mode='sum')

        bn4a = BatchNormalization(axis=1, epsilon=0.000001)(merged4a)
        relu4a = PReLU()(bn4a)
        conv4b = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv4a.get_weights())(relu4a)
        merged4b = merge([conv4, conv4b], mode='sum')

        bn4b = BatchNormalization(axis=1, epsilon=0.000001)(merged4b)
        relu4b = PReLU()(bn4b)
        conv4c = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv4a.get_weights())(relu4b)
        merged4c = merge([conv4, conv4c], mode='sum')

        bn4c = BatchNormalization(axis=1, epsilon=0.000001)(merged4c)
        relu4c = PReLU()(bn4c)
        pool4 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid', data_format='channels_last')(relu4c)
        drop4 = Dropout(0.2)(pool4)

        conv5 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')(drop4)
        bn5 = BatchNormalization(axis=1, epsilon=0.000001)(conv5)
        relu5 = PReLU()(bn5)
        conv5a = Conv2D(filters=128, kernel_size=(1, 9), padding='same')
        conv5aa = conv5a(relu5)
        merged5a = merge([conv5, conv5aa], mode='sum')

        bn5a = BatchNormalization(axis=1, epsilon=0.000001)(merged5a)
        relu5a = PReLU()(bn5a)
        conv5b = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv5a.get_weights())(relu5a)
        merged5b = merge([conv5, conv5b], mode='sum')

        bn5b = BatchNormalization(axis=1, epsilon=0.000001)(merged5b)
        relu5b = PReLU()(bn5b)
        conv5c = Conv2D(filters=128, kernel_size=(1, 9), padding='same', weights=conv5a.get_weights())(relu5b)
        merged5c = merge([conv5, conv5c], mode='sum')

        bn5c = BatchNormalization(axis=1, epsilon=0.000001)(merged5c)
        relu5c = PReLU()(bn5c)
        # pool5 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='valid', data_format='channels_last')(relu5c)
        drop5 = Dropout(0.2)(relu5c)

        conv_relu = Activation('sigmoid')(drop5)

        # TODO: what is going on with this variable name?
        l1111 = Flatten()(conv_relu)
        out = Dense(a, activation='softmax')(l1111)

        return out

    # TODO: documentation
    def create_model(self, input_shape, print_summary=False, class_count = 2):
        """Create a new RCNN model instance"""

        changed_shape = (1,input_shape[1],input_shape[0])
        input_1 = Input(changed_shape)
        output = self.RCL(input_1,a)
        model = Model(inputs=input_1, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])
        self.model = model
        return model
