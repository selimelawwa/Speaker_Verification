# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:45:49 2018

@author: selem
"""
from preprocess import get_x_y_data,mfcc_on_wav,split_audio_file,get_labels,user_data_path,enroll_data_path,numpy_files_path
import numpy as np
import os


def save_user_data_as_numpy(max_len,max_len2):
    labels = get_labels(user_data_path)
    for label in labels:
    # Init mfcc vectors
        mfcc_vectors = []
        wavfiles = []
        for file in os.listdir(user_data_path + '/' + label):
            for wavfile in os.listdir(user_data_path + '/' + label + '/' + file):
                current_path = user_data_path + '/' + label + '/' + file + '/' +wavfile
                if wavfile[-3:] != 'txt':
                    wavfiles.extend([current_path])
        for wavfile in wavfiles:
            splits = split_audio_file(file_path=wavfile)
            i=0
            if len(splits) > 0:
                while i < len(splits):
                    mfcc = mfcc_on_wav(splits[i], max_len=max_len,max_len2=max_len2)
                    mfcc_vectors.append(mfcc)
                    i = i + 1
        np.save(enroll_data_path + label + '.npy', mfcc_vectors)
        
def save_random_user_data(origin_path):
    x,y = get_x_y_data(split_ratio=0.95, random_state=42,maxsamples=40,path=origin_path,mode=1)
    x = x[::2]
    x = x[:1000]
    np.save(enroll_data_path + 'not-user' + '.npy', x)


#def get_test_data(maxsamples,path,channel,feature_dim_1,feature_dim_2):
#    # Get available labels
#    labels, indices = get_labels(path)
#    
#    # Getting first arrays
#    X = np.load(path + labels[0])
#    np.random.shuffle(X)
#    if len(X)>maxsamples:
#        X = X[:maxsamples,:,:]
#    y = np.zeros(X.shape[0])
#
#    # Append all of the dataset into one single array, same goes for y
#    for i, label in enumerate(labels[1:]):
#        x = np.load(path + label)
#        np.random.shuffle(x)
#        if len(x)>maxsamples:
#            x = x[:maxsamples,:,:]
#        X = np.vstack((X, x))
#        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))
#
#    assert X.shape[0] == len(y)
#    
#    X = X.reshape(X.shape[0], feature_dim_1, feature_dim_2, channel)
#    
#    return X,y

