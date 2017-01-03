from keras import backend as K
import os
import time
import h5py
import sys
from tagger_net import MusicTaggerCRNN
import librosa
import audio_processor as ap
import numpy as np
from keras.utils import np_utils
from math import floor
from scipy.misc import imsave
import cv2


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


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x = x[0]
    print x.shape
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255

    print x.shape
    x = cv2.applyColorMap(np.uint8(x), cv2.COLORMAP_PINK)
    print x.shape
    #x = x.transpose((1, 2, 0))
    print x.shape
    #x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths to set
model_name = "dropus_net_multi_gru_adam"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

train_songs_list = 'lists/train_songs_list_final.txt'
test_songs_list = 'lists/test_songs_gtzan_list.txt'

model = MusicTaggerCRNN(weights='msd', input_tensor=(1, 96, 1366))

model.load_weights(weights_path+model_name+'_epoch_40.h5')
for i in range(0,6):
    pop_layer(model)


model.summary()

#X_test, y_test = load_dataset('music_dataset/music_dataset_multiframe_test.h5')

img_width, img_height = 1366, 96


# Vis
layer_name = 'conv1'
filter_index = 2  # can be any integer from 0 to 511, as there are 512 filters in that layer

first_layer = get_output_layer(model, layer_name)
input_img = first_layer.input

# build a loss function that maximizes the activation
# of the nth filter of the layer considered

layer_output = get_output_layer(model, layer_name)
loss = K.mean(layer_output.output[:, filter_index, :, :])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img, K.learning_phase()], [loss, grads])


# we start from a gray image with some noise
input_img_data = np.random.random((1, 1, 96, 1366)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data, 0])
    input_img_data += grads_value * i

img = input_img_data[0]
img = deprocess_image(img)
print img.shape
imsave('%s_filter_%d.png' % (layer_name, filter_index), img)