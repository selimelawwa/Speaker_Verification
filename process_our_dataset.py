# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:53:53 2018

@author: selem
"""
import array
import os
import numpy as np
from preprocess import plot_audio_signal,open_audio_file,split_audio_file,get_labels,save_our_data_as_numpy_array,big_numpy_files_path
from pydub import AudioSegment



our_data_set_path = "D:/datasets/ourdataset"
my_data_set_path = "D:/datasets/mydataset"
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
            
def get_size_in_mega_bytes(path):
    size_in_bytes = os.stat(path).st_size
    #divide by 1024^2 to get in MB
    return size_in_bytes/(1024*1024)

def add_noise(path,label,file_name,index):
    file_path = path + '/' + label + '/' + file_name
    dest = path + '/' + label + '/' + label + '_noise' + str(index) +'.mp3'
    sound_file = open_audio_file(file_path)
    sound_numpy = np.array(sound_file.get_array_of_samples())
    #Generate White noise
    upperbound = int(np.std(sound_numpy)/20)
    wn = np.random.randint(low=0, high=upperbound, size=len(sound_numpy))
    sound_numpy_wn = sound_numpy + wn
#    print(sound_file.array_type)
    new_sound_array = array.array(sound_file.array_type, sound_numpy_wn)
    new_sound = sound_file._spawn(new_sound_array)
    newfile = new_sound.export(format='mp3',out_f=dest)
    
    
def skip_n_seconds(path,label,file_name,index,n):
    n_seconds = n * 1000
    file_path = path + '/' + label + '/' + file_name
    sound = open_audio_file(file_path)
    dest = path + '/' + label + '/' + 'sliced_' + str(index) +'.mp3'
    new_sound = sound[n_seconds:]
    newfile = new_sound.export(format='mp3',out_f=dest)

def speed_up_audio(path,label,file_name,index,speed):
    file_path = path + '/' + label + '/' + file_name
    sound = open_audio_file(file_path)
    dest = path + '/' + label + '/' + 'speedx' +str(speed)+ '_' + str(index) +'.mp3'
    new_sound = sound.speedup(playback_speed=speed, chunk_size=150, crossfade=25)
    newfile = new_sound.export(format='mp3',out_f=dest)
    
def fade_audio(path,label,file_name,index,fade):
    file_path = path + '/' + label + '/' + file_name
    sound = open_audio_file(file_path)
    dest = path + '/' + label + '/' + 'faded_' + str(index) +'.mp3'
    new_sound = sound.fade(to_gain=fade,start=650,end=5000)
    newfile = new_sound.export(format='mp3',out_f=dest)


def convert_to_mp3(path,label,file_name,index):
    file_path = path + '/' + label + '/' + file_name
    sound = open_audio_file(file_path)
    dest = path + '/' + label + '/' + 'converted_' + str(index) +'.mp3'
    newfile = sound.export(format='mp3',out_f=dest)
    os.remove(file_path)
    
def compress_audio_files(path):
    labels = get_labels(path)
    for label in labels:
        print("Compressing... ",label)
        audiofiles = []
        for audiofile in os.listdir(path + '/' + label):
            #current_path = path + '/' + label + '/' +audiofile
            audiofiles.extend([audiofile])
        i=1
        for audiofile in audiofiles:          
#            file_size = get_size_in_mega_bytes(path+"/"+label+"/"+audiofile)
#            if file_size > 0.45:
            convert_to_mp3(path=path,label=label,file_name=audiofile,index=i)
            i+=1

def create_extra_data(path):
#    compress_audio_files(path)
    labels = get_labels(path)
    for label in labels:
        print("Creating exta data for:",label)
        audiofiles = []
        for audiofile in os.listdir(path + '/' + label):
            #current_path = path + '/' + label + '/' +audiofile
            audiofiles.extend([audiofile])
        i=1
        for audiofile in audiofiles:          
            add_noise(path=path,label=label,file_name=audiofile,index=i)
            skip_n_seconds(path=path,label=label,file_name=audiofile,index=i,n=2)
            speed_up_audio(path=path,label=label,file_name=audiofile,index=i,speed=1.2)            
            i+=1



#save_our_data_as_numpy_array(max_len = 80, max_len2 = 40,origin_path=my_data_set_path,destination_path=big_numpy_files_path)
#labels = get_labels(big_numpy_files_path)
#for label in labels:
#    x = np.load(big_numpy_files_path + '/' + label)
#    if x.shape[0]<150:
#        print(label, "Shape:",x.shape[0])
        


#path = our_data_set_path
#labels = get_labels(path)
#for label in labels:
#    print("Creating exta data for:",label)
#    audiofiles = []
#    for audiofile in os.listdir(path + '/' + label):
#        #current_path = path + '/' + label + '/' +audiofile
#        audiofiles.extend([audiofile]) 
#    index = np.random.randint(low=2,high=3)
#    audiofiles = audiofiles[::index]
#    i=1
#    for audiofile in audiofiles:
#        f = np.random.randint(low=0,high=2)
#        fade_audio(path=path,label=label,file_name=audiofile,index=i,fade=f)
#        i+=1



#path = our_data_set_path
#labels = get_labels(path)
#for label in labels:
##    print("Creating exta data for:",label)
#    audiofiles = []
#    for audiofile in os.listdir(path + '/' + label):
#        #current_path = path + '/' + label + '/' +audiofile
#        audiofiles.extend([audiofile]) 
#    i=20
#    for audiofile in audiofiles:
#        if audiofile[-10:-5] == 'noise':
#            skip_n_seconds(path=path,label=label,file_name=audiofile,index=i,n=2)
#        elif audiofile[:6] == 'sliced':
#            speed_up_audio(path=path,label=label,file_name=audiofile,index=i,speed=1.25)         
#        elif audiofile[:6] == 'speedx':
#            add_noise(path=path,label=label,file_name=audiofile,index=i)
#        i+=1    





#print_user_files(our_data_set_path)