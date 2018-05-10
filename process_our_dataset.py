# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:53:53 2018

@author: selem
"""
import array
import os
import numpy as np
import random
import itertools
from playsound import playsound
from preprocess import plot_audio_signal,open_audio_file,split_audio_file,get_labels
from pydub import AudioSegment
from tqdm import tqdm


our_data_set_path = "D:/datasets/ourdataset"
#DELETE IMAGES FROM DATASET TO LEAVE ONLY AUDIOFILES
def delete_images(path):
    labels = os.listdir(path)
    for label in labels:
        for file in os.listdir(path + '/' + label):
            if file[-4:] == 'jpeg':
                os.remove(path + '/' + label + '/' +file)
            if file[-3:] == 'jpg':
                os.remove(path + '/' + label + '/' +file)
            if file[-3:] == 'JPG':
                os.remove(path + '/' + label + '/' +file)
            if file[-3:] == 'png':
                os.remove(path + '/' + label + '/' +file)
#Print all names of audiofiles of each user
def print_user_files(path):
    labels = os.listdir(path)
    for label in labels:
        print('======')
        print(label)
        print('======')
        for file in os.listdir(path + '/' + label):
            print(file)

def add_noise(path,label,file_name,index):
    file_path = path + '/' + label + '/' + file_name
    dest = path + '/' + label + '/' + label + '_noise' + str(index) +'.wav'
    sound = open_audio_file(file_path)
    sound_numpy = np.array(sound.get_array_of_samples())
    #Generate White noise
    upperbound = int(np.std(sound_numpy)/20)
    wn = np.random.randint(low=0, high=upperbound, size=len(sound_numpy))
    sound_numpy_wn = sound_numpy + wn
    print(sound.array_type)
    new_sound_array = array.array(sound.array_type, sound_numpy_wn)
    new_sound = sound._spawn(new_sound_array)
    newfile = new_sound.export(format='wav',out_f=dest)
    

def create_extra_data(path):
    labels = get_labels(path)
    for label in labels:
        audiofiles = []
        for audiofile in os.listdir(path + '/' + label):
            #current_path = path + '/' + label + '/' +audiofile
            audiofiles.extend([audiofile])
        i=1
        for audiofile in tqdm(audiofiles, "Saving vectors of label - '{}'".format(label)):
            add_noise(path=path,label=label,file_name=audiofile,index=i)
            i+=1

create_extra_data(our_data_set_path)


#delete_images(our_data_set_path)
#print_user_files(our_data_set_path)