import gumpy
import numpy as np
from datetime import datetime
import kapre
import keras
import keras.utils as ku


def load_preprocess_data(data, debug, lowcut, highcut, w0, Q, anti_drift, class_count, cutoff, axis, fs):
    """Load and preprocess data.

    The routine loads data with the use of gumpy's Dataset objects, and
    subsequently applies some post-processing filters to improve the data.
    """
    # TODO: improve documentation

    data_loaded = data.load()

    if debug:
        print('Band-pass filtering the data in frequency range from %.1f Hz to %.1f Hz... '
          %(lowcut, highcut))

        data_notch_filtered = gumpy.signal.notch(data_loaded.raw_data, cutoff, axis)
        data_hp_filtered = gumpy.signal.butter_highpass(data_notch_filtered, anti_drift, axis)
        data_bp_filtered = gumpy.signal.butter_bandpass(data_hp_filtered, lowcut, highcut, axis)

        # Split data into classes.
        # TODO: as soon as gumpy.utils.extract_trails2 is merged with the
        #       regular extract_trails, change here accordingly!
        class1_mat, class2_mat = gumpy.utils.extract_trials2(data_bp_filtered, data_loaded.trials,
                                                             data_loaded.labels, data_loaded.trial_total,
                                                             fs, nbClasses = 2)

        # concatenate data for training and create labels
        x_train = np.concatenate((class1_mat, class2_mat))
        labels_c1 = np.zeros((class1_mat.shape[0], ))
        labels_c2 = np.ones((class2_mat.shape[0], ))
        y_train = np.concatenate((labels_c1, labels_c2))

        # for categorical crossentropy
        y_train = ku.to_categorical(y_train)

        print("Data loaded and processed successfully!")
        return x_train, y_train


def print_version_info():
    now = datetime.now()

    print('%s/%s/%s' % (now.year, now.month, now.day))
    print('Keras version: {}'.format(keras.__version__))
    if keras.backend._BACKEND == 'tensorflow':
        import tensorflow
        print('Keras backend: {}: {}'.format(keras.backend._backend, tensorflow.__version__))
    else:
        import theano
        print('Keras backend: {}: {}'.format(keras.backend._backend, theano.__version__))
    print('Keras image dim ordering: {}'.format(keras.backend.image_dim_ordering()))
    print('Kapre version: {}'.format(kapre.__version__))
