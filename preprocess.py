# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 11:57:25 2018

@author: selem
"""
import os
from sklearn.model_selection import train_test_split
import numpy as np
from keras import utils
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt
from python_speech_features import mfcc as mfcc1
import soundfile as sf

path = "D:/Python projects/Speaker_Recognition/data2/dev-clean"

def get_labels():
    #get folder name 'label' for each speaker
    labels = os.listdir(path)
    #create an array from of value 0,1,2...number_ofspeakers
    label_indices = np.arange(0, len(labels))
    return labels, label_indices

def mfcc_operation(wavfile_path,max_len):
#    wav , samplingrate = librosa.load(wavfile_path, mono=True, sr=None)
    with open(wavfile_path, 'rb') as f:
        wav, samplerate = sf.read(f)
    wav = wav[::2]
    #using Librosa MFCC
    #mfcc = librosa.feature.mfcc(wav, sr=8000,n_mfcc=48,n_fft=512, hop_length=512)
    mfcc = mfcc1(wav,samplerate =8000, numcep=48,nfilt =48)
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    if (max_len < mfcc.shape[0]):
         mfcc = mfcc[:max_len,:]
    else:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return mfcc

def plot_audio_signal(wavfile_path):
    wav , sampling_rate = librosa.load(wavfilepath, mono=True, sr=None)
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of ')
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sampling_rate/len(wav), sampling_rate), wav)
    
def save_data_as_numpy_array(max_len):
    
    labels, _ = get_labels()
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []
        wavfiles = []
#        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for file in os.listdir(path + '/' + label):
            for wavfile in os.listdir(path + '/' + label + '/' + file):
                current_path = path + '/' + label + '/' + file + '/' +wavfile
                if wavfile[-3:] != 'txt':
                    wavfiles.extend([current_path])
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = mfcc_operation(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.85, random_state=42):
    # Get available labels
    labels, indices = get_labels()

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



#mymfcc = mfcc_operation(wavfile_path = 'D:/Python projects/Speaker_Recognition/test/6313-76958-0009.flac',max_len=48)
#print(mymfcc.shape)
#transpose = np.transpose(mymfcc)
#
#plt.figure(figsize=(12, 4))
#librosa.display.specshow(mymfcc)
#plt.ylabel('MFCC coeffs')
#plt.xlabel('Time')
#plt.title('MFCC')
#plt.colorbar()
#plt.tight_layout()



