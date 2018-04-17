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

#path = "D:/Python projects/Speaker_Recognition/mydata/train-clean-100"

numpyfilespath = "D:/Python projects/Speaker_Recognition/numpydata/"

enrolldata = "D:/Python projects/Speaker_Recognition/enroll-data/"
userdatapath = "D:/Python projects/Speaker_Recognition/userdata/"


path = "D:/datasets/LibriSpeech/train-clean-100"

#we want variables to have almost zero mean and equal variance

def get_labels(path):
    #get folder name 'label' for each speaker
    labels = os.listdir(path)
    #create an array from of value 0,1,2...number_ofspeakers
    label_indices = np.arange(0, len(labels))
    return labels, label_indices


def index_of_speech_start_updated(wav):
    i=0
    while i<len(wav):
        if wav[i] < 0.004:
            i = i + 1
        else: break
    return i
               
            
def split_audio_file(file_path):
    sound = AudioSegment.from_file(file_path, format="flac")
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
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    if (max_len2 > mfcc.shape[1]):
        pad_width = max_len2 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len2]
    return mfcc

def plot_audio_signal(wav, sampling_rate,title):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of '+ title)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, len(wav), len(wav)), wav)

def save_data_as_numpy_array(max_len,max_len2):
    
    labels, _ = get_labels(path)
    for label in labels:
        if os.path.exists(numpyfilespath + label + '.npy'):
            print(label, 'exists...skipping')
        else:
        # Init mfcc vectors
            mfcc_vectors = []
            wavfiles = []
            for file in os.listdir(path + '/' + label):
                for wavfile in os.listdir(path + '/' + label + '/' + file):
                    current_path = path + '/' + label + '/' + file + '/' +wavfile
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
            np.save(numpyfilespath + label + '.npy', mfcc_vectors)
            
def save_user_data_as_numpy(max_len,max_len2):
    
    labels, _ = get_labels(userdatapath)
    for label in labels:
    # Init mfcc vectors
        mfcc_vectors = []
        wavfiles = []
        for file in os.listdir(userdatapath + '/' + label):
            for wavfile in os.listdir(userdatapath + '/' + label + '/' + file):
                current_path = userdatapath + '/' + label + '/' + file + '/' +wavfile
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
        np.save(enrolldata + label + '.npy', mfcc_vectors)
        

def get_train_test(split_ratio, random_state,maxsamples,path):
    # Get available labels
    labels, indices = get_labels(path)
    
    # Getting first arrays
    #X = np.load(numpyfilespath + labels[0] + '.npy')
    X = np.load(path + labels[0])
    np.random.shuffle(X)
    if len(X)>maxsamples:
        X = X[:maxsamples,:,:]
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        #x = np.load(numpyfilespath + label + '.npy')
        x = np.load(path + label)
        np.random.shuffle(x)
        if len(x)>maxsamples:
            x = x[:maxsamples,:,:]
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)
    
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)


def save_random_user_data():
    t,x,t2,y = get_train_test(split_ratio=0.8, random_state=42,maxsamples=40,path=numpyfilespath)
    x = x[:300]
    np.save(enrolldata + '0' + '.npy', x)




#def mfcc_operation(wavfile_path,max_len,max_len2):
##    wav , samplingrate = librosa.load(wavfile_path, mono=True, sr=None)
#    with open(wavfile_path, 'rb') as f:
#        wav, samplerate = sf.read(f)
#    index = index_of_speech_start_updated(wav)
#    wav = wav[index:]
#    #using Librosa MFCC
#    #mfcc = librosa.feature.mfcc(wav, sr=8000,n_mfcc=48,n_fft=512, hop_length=512)
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



#mymfcc = mfcc_operation(wavfile_path = 'D:/Python projects/Speaker_Recognition/test/8842-302203-0008.flac',max_len=80,max_len2=40)
#print(mymfcc.shape)
#transpose because , librosa display does a transform
#transpose = np.transpose(mymfcc)

#for file in os.listdir('D:/Python projects/Speaker_Recognition/data2/dev-clean/174/50561'):
#    file_path = 'D:/Python projects/Speaker_Recognition/data2/dev-clean/174/50561/' + file
#    tempmfcc = mfcc_operation(wavfile_path = file_path,max_len=80,max_len2=40)
#    transpose = np.transpose(tempmfcc)
#    plt.figure(figsize=(10, 5))
#    librosa.display.specshow(transpose)
#    plt.ylabel('MFCC coeffs')
#    plt.xlabel('Time')
#    plt.title('MFCC')
#    plt.colorbar()
#    plt.tight_layout()


