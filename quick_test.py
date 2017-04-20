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
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, extract_melgrams
import matplotlib.pyplot as plt

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


# Initialize model
model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if LOAD_WEIGHTS:
    model.load_weights(weights_path+'crnn_net_gru_adam_ours_epoch_40.h5')

#model.summary()

X_test, num_frames_test= extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre='')

num_frames_test = np.array(num_frames_test)

t0 = time.time()

print '\n--------- Predicting ---------','\n'

results = np.zeros((X_test.shape[0], tags.shape[0]))
predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
predicted_labels_frames = np.zeros((X_test.shape[0], 1))

song_paths = open(test_songs_list, 'r').read().splitlines()

previous_numFrames = 0
n=0
for i in range(0, num_frames_test.shape[0]):
    print 'Song number' +str(i)+ ': ' + song_paths[i]

    num_frames=num_frames_test[i]
    print 'Num_frames of 30s: ', str(num_frames),'\n'

    results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
        X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])

    s_counter = 0
    for j in range(previous_numFrames, previous_numFrames+num_frames):
        #normalize the results
        total = results[j,:].sum()
        results[j,:]=results[j,:]/total
        print 'Percentage of genre prediction for seconds '+ str(20+s_counter*30) + ' to ' \
            + str(20+(s_counter+1)*30) + ': '
        sort_result(tags, results[j,:].tolist())

        predicted_label_frames=predict_label(results[j,:])
        predicted_labels_frames[n]=predicted_label_frames
        s_counter += 1
        n+=1

    print '\n', 'Mean genre of the song: '
    results_song = results[previous_numFrames:previous_numFrames+num_frames]

    mean=results_song.mean(0)
    sort_result(tags, mean.tolist())

    predicted_label_mean=predict_label(mean)

    predicted_labels_mean[i]=predicted_label_mean
    print '\n','The predicted music genre for the song is', str(tags[predicted_label_mean]),'!\n'

    previous_numFrames = previous_numFrames+num_frames

    print '************************************************************************************************'

colors = ['b','g','c','r','m','k','y','#ff1122','#5511ff','#44ff22']
fig, ax = plt.subplots()
index = np.arange(tags.shape[0])
opacity = 1
bar_width = 0.2
print mean
#for g in range(0, tags.shape[0]):
plt.bar(left=index, height=mean, width=bar_width, alpha=opacity, color=colors)

plt.xlabel('Genres')
plt.ylabel('Percentage')
plt.title('Scores by genre')
plt.xticks(index + bar_width / 2, tags)
plt.tight_layout()
fig.autofmt_xdate()
plt.savefig('genres_prediction.png')

