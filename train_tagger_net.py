from keras import backend as K
import os
import time
import h5py
import sys
from tagger_net import MusicTaggerCRNN
from keras.layers import Dense
from keras.optimizers import SGD
import librosa
import audio_processor as ap
import numpy as np
from keras.utils import np_utils
from math import floor
from music_tagger_cnn import MusicTaggerCNN
from utils import save_data, load_dataset, save_dataset, sort_results, extract_melgrams

# Parameters to set
TRAIN = 1
TEST = 0
EXTRACT_FEAT = 0

SAVE_MODEL = 0
SAVE_WEIGHTS = 0

LOAD_MODEL = 0
LOAD_WEIGHTS = 0

# Dataset
MULTIFRAMES = 1
SAVE_DB = 0
LOAD_DB = 1

# Model parameters
nb_classes = 10
nb_epoch = 40
batch_size = 100

time_elapsed = 0

# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths to set
model_name = "convnet_net_adam"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

train_songs_list = 'lists/train_songs_list_final.txt'
test_songs_list = 'lists/test_songs_gtzan_list.txt'


# Divide the song into multiple frames of 29.1s or take the center crop.
if MULTIFRAMES:
    train_gt_list = 'lists/train_gt_list_multiframes.txt'
    test_gt_list = 'lists/test_gt_list_multiframes.txt'
else:
    train_gt_list = 'lists/train_gt_list.txt'
    test_gt_list = 'lists/test_gt_list.txt'

# Create directories for the models & weights
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print 'Path created: ', model_path

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print 'Path created: ', weights_path

# Data Loading
if LOAD_DB:
    if MULTIFRAMES:
        print 'Loading dataset multiframe...'
        X_train,  y_train  = load_dataset('music_dataset/music_dataset_multiframe_train.h5')
        X_test, y_test = load_dataset('music_dataset/music_dataset_multiframe_test.h5')
    else:
        X_train, X_test, y_train, y_test = load_dataset('music_dataset/music_dataset.h5')

else:
    #
    X_train, y_train = extract_melgrams(train_songs_list, process_all_song=False, num_songs_genre=30)
    print('X_train shape:', X_train.shape)
    X_test, y_test = extract_melgrams(test_songs_list, process_all_song=True, num_songs_genre=100)

print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')

y_train = np.array(y_train)
y_test = np.array(y_test)

if SAVE_DB:
    if MULTIFRAMES:
        save_dataset('music_dataset/music_dataset_multiframe_train.h5', X_train, y_train)
        save_dataset('music_dataset/music_dataset_multiframe_test.h5', X_test,y_test)
    else:
        save_dataset('music_dataset/music_dataset.h5', X_train, X_test, y_train, y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print 'Shape labels y_train: ', Y_train.shape
print 'Shape labels y_test: ', Y_test.shape

# Initialize model
#model = MusicTaggerCRNN(weights='msd', input_tensor=(1, 96, 1366))

model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if LOAD_WEIGHTS:
    model.load_weights(weights_path+model_name+'_epoch_40.h5')

model.summary()

# Save model architecture
if SAVE_MODEL:
    json_string = model.to_json()
    f = open(model_path+model_name+".json", 'w')
    f.write(json_string)
    f.close()

# Train model
if TRAIN:
    try:
        print ("Training the model")
        f_train = open(model_path+model_name+"_scores_training.txt", 'w')
        f_test = open(model_path+model_name+"_scores_test.txt", 'w')
        f_scores = open(model_path+model_name+"_scores.txt", 'w')
        for epoch in range(1,nb_epoch+1):
            t0 = time.time()
            print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
            sys.stdout.flush()
            scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))
            time_elapsed = time_elapsed + time.time() - t0
            print ("Time Elapsed: " +str(time_elapsed))
            sys.stdout.flush()

            score_train = model.evaluate(X_train, Y_train, verbose=0)
            print('Train Loss:', score_train[0])
            print('Train Accuracy:', score_train[1])
            f_train.write(str(score_train)+"\n")

            score_test = model.evaluate(X_test, Y_test, verbose=0)
            print('Test Loss:', score_test[0])
            print('Test Accuracy:', score_test[1])
            f_test.write(str(score_test)+"\n")

            f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1]) + "\n")

            if SAVE_WEIGHTS and epoch % 5 == 0:
                model.save_weights(weights_path + model_name + "_epoch_" + str(epoch) + ".h5")
                print("Saved model to disk in: " + weights_path + model_name + "_epoch" + str(epoch) + ".h5")

        f_train.close()
        f_test.close()
        f_scores.close()

        # Save time elapsed
        f = open(model_path+model_name+"_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()

    # Save files when an sudden close happens / ctrl C
    except:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path + model_name + "_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()
    finally:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path + model_name + "_time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()

if TEST:
    t0 = time.time()
    print 'Predicting...'
    print

    test_numFrames_total = load_numframes()
    #print test_numFrames_total.shape[0]

    song_paths = open(test_songs_list, 'r').read().splitlines()

    previous_numFrames = 0
    for i in range(0, test_numFrames_total.shape[0]):
        print song_paths[i]

        num_frames=test_numFrames_total[i]
        print 'Num_frames: ', str(num_frames)
        print

        results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
            X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])

        print results[previous_numFrames:previous_numFrames+num_frames].sum()
        #print results[previous_numFrames:previous_numFrames+num_frames] * 100

        for j in range(previous_numFrames,previous_numFrames+num_frames):
            sort_result(tags, results[j,:].tolist())

        print "Mean of the song: "
        results_song = results[previous_numFrames:previous_numFrames+num_frames]

        mean=results_song.mean(0)
        #print mean
        sort_result(tags, mean.tolist())

        previous_numFrames = previous_numFrames+num_frames

        break
        print '\n\n\n'


if EXTRACT_FEAT:
    batch_size = 50
    all_descriptors = np.zeros((X_test.shape[0], 32))
    final_layer = get_layer(model, "final_drop")
    all_scores = np.zeros((num_samples, nb_classes))
    get_output = K.function([model.layers[0].input, K.learning_phase()],
                            [final_conv_layer.output, model.layers[-1].output])

    num_it = int(math.floor(num_samples / batch_size))
    last_batch = num_samples % batch_size
    batch_size_loop = batch_size

    for i in range(0, num_it+1):
        t0 = time.time()
        if i == num_it:
            if last_batch != 0:
                x = X_test[i*batch_size:batch_size*i+last_batch, :, :, :]
                batch_size_loop = last_batch
            else:
                break
        else:
            x = X_test[i*batch_size:batch_size*(i+1), :, :, :]

        print 'Batch number: ', i

        [desc_outputs, scores] = get_output([x, 0])
        all_descriptors[i*batch_size:i*batch_size+desc_outputs.shape[0], :, :, :] = desc_outputs

    save_data(all_descriptors, 'descriptors_gtzan.h5')



