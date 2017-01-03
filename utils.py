import os
import time
import h5py
import sys
import librosa
import audio_processor as ap
import numpy as np
from math import floor



# Functions Definition

def save_data(data, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)


def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
    return data, labels


def save_dataset(path, data, labels):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)


def load_numframes():
    with open('lists/test_numframes.txt', "r") as insTest:
        test_numFrames_total = []
        for lineTest in insTest:
            test_numFrames_total.append(int(lineTest))
        test_numFrames_total = np.array(test_numFrames_total)
        # print test_numFrames_total

    return test_numFrames_total


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print name, ':', '%5.3f  ' % score, '   ',
    print


# Melgram computation
def extract_melgrams(list_path, process_all_song, num_songs_genre):
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    song_paths = open(list_path, 'r').read().splitlines()
    labels = list()
    for song_ind, song_path in enumerate(song_paths):
        print song_path
        if MULTIFRAMES:
            melgram = ap.compute_melgram_multiframe(song_path, process_all_song)
            num_frames = melgram.shape[0]
            print 'num frames:', num_frames
            index = int(floor(song_ind/num_songs_genre))
            for i in range(0, num_frames):
                labels.append(index)
        else:
            melgram = ap.compute_melgram(song_path)

        melgrams = np.concatenate((melgrams, melgram), axis=0)

    return melgrams, labels