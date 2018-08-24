# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:42:37 2018

@author: selem
"""

from preprocess import *
import keras  
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import load_model
from keras import regularizers
import numpy as np
from keras.callbacks import EarlyStopping
from keras import metrics
import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from importance_sampling.training import ImportanceTraining

epochs = 50
batch_size = 100
verbose = 1
labels= get_labels(big_numpy_files_path)
num_classes = len(labels)
 
#save_data_as_numpy_array(max_len = feature_dim_1, max_len2 = feature_dim_2,origin_path=data_set_path,destination_path=numpy_files_path)

#Get X & Y 
X_train, X_valid, X_test ,y_train,y_valid,y_test = get_x_y_data(split_ratio=0.8, random_state=42,maxsamples=20,path=big_numpy_files_path)
X_train, X_valid, X_test = reshape_prepare_for_input(X1=X_train, X2=X_valid, X3=X_test,channel=1)
#Dimensions
dim_1 = X_train.shape[1]
dim_2 = X_train.shape[2]
dim_3 = X_train.shape[3]

#one hot encoding of Y
y_train_hot = to_categorical(y_train)
y_valid_hot = to_categorical(y_valid)
y_test_hot = to_categorical(y_test)

#Model 
model = Sequential()
model.add(Conv2D(411, kernel_size=(6, 6),strides=2,use_bias=True, activation='relu', input_shape=(dim_1, dim_2,dim_3)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(128, activation='relu', use_bias=True))
model.add(Dropout(0.35))
model.add(Dense(128, activation='relu',use_bias=True))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu',use_bias=True))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loss=keras.losses.categorical_crossentropy

model.compile(loss=loss,optimizer=sgd,metrics=['accuracy'])


def predict(sample, model):
#    sample = mfcc_operation(wavfile_path = filepath,max_len = feature_dim_2)
    sample_reshaped = sample.reshape(1, dim_1,dim_2,dim_3)
    return get_labels(big_numpy_files_path)[np.argmax(model.predict(sample_reshaped))]
	

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=verbose, mode='auto')

ImportanceTraining(model).fit(
   X_train, y_train_hot,
    batch_size=batch_size,
    epochs=epochs,
    verbose=verbose,
    validation_data=(X_valid, y_valid_hot)
)

#model.fit(X_train, y_train_hot,batch_size=batch_size,callbacks=[earlystopping],epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid_hot))

#model.save('firstbigmodel.h5') 

