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
TRAIN = 0
TEST = 1

SAVE_MODEL = 0
SAVE_WEIGHTS = 0

LOAD_MODEL = 0
LOAD_WEIGHTS = 1

# Dataset
MULTIFRAMES = 1
SAVE_DB = 0
LOAD_DB = 0

# Model parameters
nb_classes = 10
nb_epoch = 40
batch_size = 100

time_elapsed = 0

# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths to set
model_name = "crnn_net_adam_ours"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

train_songs_list = 'lists/train_songs_list_ours.txt'
test_songs_list = 'lists/test_songs_list_ours.txt'
#test_songs_list = 'lists/test_songs_gtzan_list.txt'

# Create directories for the models & weights
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print 'Path created: ', model_path

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print 'Path created: ', weights_path

# Divide the song into multiple frames of 29.1s or take the center crop.
if MULTIFRAMES:
    train_gt_list = 'lists/train_gt_list_multiframes.txt'
    test_gt_list = 'lists/test_gt_ours.txt'
else:
    train_gt_list = 'lists/train_gt_list.txt'
    test_gt_list = 'lists/test_gt_list.txt'

# Data Loading

if LOAD_DB:
    if MULTIFRAMES:
        print 'Loading dataset multiframe...'
        X_train,  y_train, num_frames_train  = load_dataset('')
        X_test, y_test, num_frames_test = load_dataset('')
    else:
        X_train, X_test, y_train, y_test = load_dataset('')

# Compute mel-spectogram for all the frames
else:
    X_train, y_train, num_frames_train = extract_melgrams(train_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre=20)
    print('X_train shape:', X_train.shape)
    X_test, y_test, num_frames_test = extract_melgrams(test_songs_list, MULTIFRAMES, process_all_song=False, num_songs_genre=10)


print(X_train.shape, 'train samples')
print(X_test.shape, 'test samples')


y_train = np.array(y_train)
y_test = np.array(y_test)

if SAVE_DB:
    if MULTIFRAMES:
        save_dataset('music_dataset/music_dataset_multiframe_train.h5', X_train, y_train,num_frames_train)
        save_dataset('music_dataset/music_dataset_multiframe_test.h5', X_test,y_test,num_frames_test)
    else:
        save_dataset('music_dataset/music_dataset.h5', X_train, X_test, y_train, y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print 'Shape labels y_train: ', Y_train.shape
print 'Shape labels y_test: ', Y_test.shape



# Initialize model
model = MusicTaggerCRNN(weights='msd', input_tensor=(1, 96, 1366))
#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366))
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
    print 'Predicting...','\n'

    real_labels_mean = load_gt(test_gt_list)
    real_labels_frames = y_test

    results = np.zeros((X_test.shape[0], tags.shape[0]))
    predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
    predicted_labels_frames = np.zeros((y_test.shape[0], 1))


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

        if predicted_label_mean != real_labels_mean[i]:
            print 'WRONG!!'


        previous_numFrames = previous_numFrames+num_frames

        #break
        print '\n\n\n'

    cnf_matrix_frames = confusion_matrix(real_labels_frames, predicted_labels_frames)
    plot_confusion_matrix(cnf_matrix_frames, classes=tags, title='Confusion matrix (frames)')

    cnf_matrix_mean = confusion_matrix(real_labels_mean, predicted_labels_mean)
    plot_confusion_matrix(cnf_matrix_mean, classes=tags, title='Confusion matrix (using mean)')






