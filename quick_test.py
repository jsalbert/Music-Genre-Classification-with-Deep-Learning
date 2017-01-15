from keras import backend as K
import os
import time
import h5py
import sys
from tagger_net import MusicTaggerCRNN
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from music_tagger_cnn import MusicTaggerCNN
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, load_gt, plot_confusion_matrix, extract_melgrams

# Parameters to set
TEST = 1

LOAD_MODEL = 0
LOAD_WEIGHTS = 1
MULTIFRAMES = 1
time_elapsed = 0

# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths to set
model_name = "example_model"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"


test_songs_list = 'list_example.txt'
# Data Loading

X_test, num_frames_test= extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')
#print X_test.shape
#print num_frames_test.shape

num_frames_test = np.array(num_frames_test)
# Initialize model
model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))
#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if LOAD_WEIGHTS:
    model.load_weights(weights_path+'crnn_net_gru_adam_ours_epoch_40.h5')

#model.summary()


t0 = time.time()
print 'Predicting...','\n'

results = np.zeros((X_test.shape[0], tags.shape[0]))
predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
predicted_labels_frames = np.zeros((X_test.shape[0], 1))

song_paths = open(test_songs_list, 'r').read().splitlines()

previous_numFrames = 0
n=0
for i in range(0, num_frames_test.shape[0]):
    print song_paths[i]

    num_frames=num_frames_test[i]
    print 'Num_frames: ', str(num_frames),'\n'

    results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
        X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])


    for j in range(previous_numFrames,previous_numFrames+num_frames):
        #normalize the results
        total = results[j,:].sum()
        results[j,:]=results[j,:]/total
        sort_result(tags, results[j,:].tolist())

        predicted_label_frames=predict_label(results[j,:])
        predicted_labels_frames[n]=predicted_label_frames
        n+=1


    print '\n',"Mean of the song: "
    results_song = results[previous_numFrames:previous_numFrames+num_frames]

    mean=results_song.mean(0)
    sort_result(tags, mean.tolist())

    predicted_label_mean=predict_label(mean)

    predicted_labels_mean[i]=predicted_label_mean
    print '\n','Predicted label: ', str(tags[predicted_label_mean]),'\n'

    previous_numFrames = previous_numFrames+num_frames

    print '\n\n\n'

#cnf_matrix_frames = confusion_matrix(real_labels_frames, predicted_labels_frames)
#plot_confusion_matrix(cnf_matrix_frames, classes=tags, title='Confusion matrix (frames)')

#cnf_matrix_mean = confusion_matrix(real_labels_mean, predicted_labels_mean)
#plot_confusion_matrix(cnf_matrix_mean, classes=tags, title='Confusion matrix (using mean)')






