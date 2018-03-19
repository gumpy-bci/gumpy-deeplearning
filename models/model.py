import os
from abc import ABC, abstractmethod
import keras
from keras.models import model_from_json
from keras.callbacks import CSVLogger, ModelCheckpoint


class Model(ABC):
    """An abstract deep learning model.

    The abstract class functions as a facade for the backend. Although
    gumpy-deeplearning currently uses keras, it is possible that future releases
    may use different front- or backends. The Model ABC should represent the
    baseline for any such model.

    For more information about the reason behind ``Model``, see https://xkcd.com/927/

    """

    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def from_json(self):
        pass




class KerasModel(Model):
    """ABC for Models that rely on keras.

    The ABC provides an implementation to generate callbacks to monitor the
    model and write the data to HDF5 files. The function ``fit`` simply forwards
    to the keras' ``fit``, but will enable monitoring if wanted.

    """

    def __init__(self, name):
        super(KerasModel, self).__init__(name)
        self.callbacks = None


    def get_callbacks(self):
        """Returns callbacks to monitor the model.

        """

        # save weights in an HDF5 file
        model_file = self.name + '_monitoring' + '.h5'
        checkpoint = ModelCheckpoint(model_file, monitor = 'val_loss',
                                     verbose = 0, save_best_only = True, mode = 'min')
        log_file = self.name + '.csv'
        csv_logger = CSVLogger(log_file, append = True, separator = ';')
        callbacks_list = [csv_logger] # callback list

        self.callbacks = callbacks_list
        return callbacks_list


    def fit(self, x, y, monitor=True, **kwargs):
        # TODO: allow user to specify filename
        if monitor and (self.callbacks is None):
            self.get_callbacks()

        if self.callbacks is not None:
            self.model.fit(x, y, **kwargs, callbacks=self.callbacks)
        else:
            self.model.fit(x, y, **kwargs)


    def evaluate(self, x, y, **kwargs):
        return self.model.evaluate(x, y, **kwargs)


    def from_json(self, model_file_name=None):
        try:
            # set the model_file_name if it is not passed to the function
            if model_file_name is None:
                model_file_name = self.name

            # load trained model
            model_path = model_file_name + ".json"
            if not os.path.isfile(model_path):
                raise IOError('file "%s" does not exist' %(model_path))
            model = model_from_json(open(model_path).read())

            # load weights of trained model
            model_weight_path = model_file + ".hdf5"
            if not os.path.isfile(model_weight_path):
                raise OSError('file "%s" does not exist' %(model_path))
            model.load_weights(model_weight_path)

            return model
        except IOError:
            print(IOError)
            return None
