import os
import time
import h5py
import sys
import librosa
import audio_processor as ap
import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import floor
from operator import truediv



# Functions Definition

def save_data(data, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)


def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
        num_frames = np.array(hf.get('num_frames'))
    return data, labels, num_frames


def save_dataset(path, data, labels, num_frames):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('num_frames', data=num_frames)


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print name, ':', '%5.3f  ' % score, '   ',
    print


def predict_label(preds):
    labels=preds.argsort()[::-1]
    return labels[0]


def load_gt(path):
    with open(path, "r") as insTest:
        gt_total = []
        for lineTest in insTest:
            gt_total.append(int(lineTest))
        gt_total = np.array(gt_total)
        # print test_numFrames_total

    return gt_total


def plot_confusion_matrix(cnf_matrix, classes, title):

    cnfm_suma=cnf_matrix.sum(1)
    cnfm_suma_matrix = np.repeat(cnfm_suma[:,None],cnf_matrix.shape[1],axis=1)

    cnf_matrix=10000*cnf_matrix/cnfm_suma_matrix
    cnf_matrix=cnf_matrix/(100*1.0)
    print cnf_matrix

    #print map(truediv,cnf_matrix, cnfm_suma_matrix)

    fig=plt.figure()
    cmap=plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #plt.show()
    fig.savefig(title)



# Melgram computation
def extract_melgrams(list_path, MULTIFRAMES, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    song_paths = open(list_path, 'r').read().splitlines()
    labels = list()
    num_frames_total = list()
    for song_ind, song_path in enumerate(song_paths):
        print song_path
        if MULTIFRAMES:
            melgram = ap.compute_melgram_multiframe(song_path, process_all_song)
            num_frames = melgram.shape[0]
            num_frames_total.append(num_frames)
            print 'num frames:', num_frames
            if num_songs_genre != '':
                index = int(floor(song_ind/num_songs_genre))
                for i in range(0, num_frames):
                    labels.append(index)
            else:
                pass
        else:
            melgram = ap.compute_melgram(song_path)

        melgrams = np.concatenate((melgrams, melgram), axis=0)
    if num_songs_genre != '':
        return melgrams, labels, num_frames_total
    else:
        return melgrams, num_frames_total
