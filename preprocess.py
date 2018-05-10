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
from pydub import AudioSegment
from pydub.silence import *


#data specs
feature_dim_1 = 80  #time
feature_dim_2 = 40  #frequency
channel = 1

#numpy
numpy_files_path = "D:/Python projects/Speaker_Recognition/numpy-data/"
numpy_test_path = "D:/Python projects/Speaker_Recognition/numpy-test/"
enroll_data_path = "D:/Python projects/Speaker_Recognition/numpy-enroll/"
#for time 0.8
big_numpy_files_path = "D:/Python projects/Speaker_Recognition/big-numpy-data/"

#original data
data_set_path = "D:/datasets/LibriSpeech/train-clean-100"
user_data_path = "D:/Python projects/Speaker_Recognition/userdata/"
test_data_path = "D:/Python projects/Speaker_Recognition/test"


def get_labels(path):
    #get folder name 'label' for each speaker
    labels = os.listdir(path)
    return labels

def index_of_speech_start_updated(wav):
    i=0
    while i<len(wav):
        if wav[i] < 0.004:
            i = i + 1
        else: break
    return i
               
def open_audio_file(file_path):
    if file_path[-3:] == "wav":
        sound = AudioSegment.from_wav(file_path)
    elif file_path[-4:] == "flac":
        sound = AudioSegment.from_file(file_path, format="flac")
    elif file_path[-3:] == "ogg":
        sound = AudioSegment.from_ogg(file_path)
    elif file_path[-3] == "mp4":
        sound = AudioSegment.from_file(file_path, format="mp4")
    return sound
    
def split_audio_file(file_path):
    sound = open_audio_file(file_path)
    chunks = split_on_silence(
        sound,
        # split on silences longer than 1000ms (1 sec)
        min_silence_len=500,
        # anything under -16 dBFS is considered silence
        silence_thresh=-36, 
        # keep 200 ms of leading/trailing silence
        keep_silence=100
    )
    listofchunks = list()
    i=0
    while i < len(chunks):
        temp = np.array(chunks[i].get_array_of_samples())
        listofchunks.append(temp)
        i = i + 1
    return listofchunks

def mfcc_on_wav(wav,max_len,max_len2):
    index = index_of_speech_start_updated(wav)
    wav = wav[index:]
    mfcc = mfcc1(wav,samplerate=16000, numcep=48,nfilt =48)
    #If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len < mfcc.shape[0]):
         mfcc = mfcc[:max_len,:]
    else:
        pad_width = max_len - mfcc.shape[0]
        pad_width = int(pad_width)
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    if (max_len2 > mfcc.shape[1]):
        pad_width = max_len2 - mfcc.shape[1]
        pad_width = int(pad_width)
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len2]
    return mfcc

def plot_audio_signal(wav, title, sampling_rate=16000):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of '+ title)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, len(wav), len(wav)), wav)
    

def save_data_as_numpy_array(max_len,max_len2,origin_path,destination_path):
    labels = get_labels(origin_path)
    for label in labels:
        if os.path.exists(destination_path + label + '.npy'):
            print(label, 'exists...skipping')
        else:
        # Init mfcc vectors
            mfcc_vectors = []
            wavfiles = []
            for file in os.listdir(origin_path + '/' + label):
                for wavfile in os.listdir(origin_path + '/' + label + '/' + file):
                    current_path = origin_path + '/' + label + '/' + file + '/' +wavfile
                    if wavfile[-3:] != 'txt':
                        wavfiles.extend([current_path])
            for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
                splits = split_audio_file(file_path=wavfile)
                i=0
                if len(splits) > 0:
                    while i < len(splits):
                        
                        mfcc = mfcc_on_wav(splits[i], max_len=max_len,max_len2=max_len2)
                        mfcc_vectors.append(mfcc)
                        i = i + 1
            np.save(destination_path + label + '.npy', mfcc_vectors)
                   

def get_x_y_data(maxsamples,path,split_ratio=0.8, random_state=42,mode=None):
    # Get available labels
    labels = get_labels(path)
    # Getting first arrays
    X = np.load(path + labels[0])
    np.random.shuffle(X)
    if len(X)>maxsamples:
        X = X[:maxsamples,:,:]
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(path + label)
        np.random.shuffle(x)
        if len(x)>maxsamples:
            x = x[:maxsamples,:,:]
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    #if mode==1 will return just X-Y 
    #else will retun X-Y {train-validation-test}
    if mode==1:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
        return X_train,y_train
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)
        return X_train, X_valid, X_test, y_train, y_valid, y_test


def reshape_prepare_for_input(X1,X2=None,X3=None,channel=1):
    #subtract mean and divide by standard deviation so input data has zero mean ( Normalization )
    dim_1 = X1.shape[1]
    dim_2 = X1.shape[2]
    std = np.std(X1)
    mean = np.mean(X1)
    X1 = X1.reshape(X1.shape[0], dim_1, dim_2, channel)
    X1 = X1 - mean
    X1 = X1 / std
    
    if(X2 is not None):
        X2 = X2.reshape(X2.shape[0], dim_1, dim_2, channel)
        X2 = X2 - mean
        X2 = X2 / std
    if(X3 is not None):
        X3 = X3.reshape(X3.shape[0], dim_1, dim_2, channel)
        X3 = X3 - mean
        X3 = X3 / std
    
    return X1, X2, X3
    
#OLD MFCC on wav
#def mfcc_on_wav(wav,max_len,max_len2):
#    index = index_of_speech_start_updated(wav)
#    wav = wav[index:]
#    mfcc = mfcc1(wav,samplerate=16000, numcep=48,nfilt =48)
#    #If maximum length exceeds mfcc lengths then pad the remaining ones
#    if (max_len < mfcc.shape[0]):
#         mfcc = mfcc[:max_len,:]
#    else:
#        pad_width = max_len - mfcc.shape[0]
#        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
#    if (max_len2 > mfcc.shape[1]):
#        pad_width = max_len2 - mfcc.shape[1]
#        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
#    # Else cutoff the remaining parts
#    else:
#        mfcc = mfcc[:, :max_len2]
#    return mfcc
