from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def MusicTaggerCRNN(weights='msd', input_tensor=None):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''

    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', trainable=False)(x)
    x = Dropout(0.1, name='dropout1', trainable=False)(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2', trainable=False)(x)
    x = Dropout(0.1, name='dropout2', trainable=False)(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', trainable=False)(x)
    x = Dropout(0.1, name='dropout3', trainable=False)(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', trainable=False)(x)
    x = Dropout(0.1, name='dropout4', trainable=False)(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3, name='final_drop')(x)

    if weights is None:
        # Create model
        x = Dense(10, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        # Load input
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        # Create model
        initial_model = Model(melgram_input, x)
        initial_model.load_weights('weights/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)

        # Eliminate last layer
        pop_layer(initial_model)

        # Add new Dense layer
        last = initial_model.get_layer('final_drop')
        preds = (Dense(10, activation='sigmoid', name='preds'))(last.output)
        model = Model(initial_model.input, preds)

        return model