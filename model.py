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



feature_dim_1 = 40
feature_dim_2 = 40
channel = 1
epochs = 75
batch_size = 100

verbose = 1

save_data_as_numpy_array(max_len = feature_dim_1, max_len2 = feature_dim_2)

labels,_ = get_labels(numpyfilespath)

   
    

#maxsamples = 0
#avg = 0
#for label in labels:
#    data = np.load(numpyfilespath + label )
#    avg = avg + data.shape[0]
#    #print('Info about label:',label)
#    print(label,' Full dataset tensor:', data.shape)
#    if data.shape[0] > maxsamples:
#        if data.shape[0] < 200:
#            maxsamples = data.shape[0]


num_classes = len(labels)


X_train, X_valid, y_train, y_valid = get_train_test(split_ratio=0.8, random_state=42,maxsamples=50,path=numpyfilespath)

X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
std = np.std(X_train)
mean = np.mean(X_train)
print('Mean: ',mean)
print('STD: ', std)
X_train = X_train - mean
X_train = X_train / std
print('after sub mean and div by std')
print('X train mean',np.mean(X_train))
print('X train std',np.std(X_train))


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

# Reshaping to perform 2D convolution

X_valid = X_valid.reshape(X_valid.shape[0], feature_dim_1, feature_dim_2, channel)
X_valid = X_valid - mean
X_valid = X_valid / std

X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test - mean
X_test = X_test / std

#one hot encoding of Y
y_train_hot = to_categorical(y_train)
y_valid_hot = to_categorical(y_valid)
y_test_hot = to_categorical(y_test)


def get_model():
    model = Sequential()
    model.add(Conv2D(128, kernel_size=(6, 6),strides=2, activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(64, kernel_size=(2, 2),strides=1, activation='relu'))
    
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.4))
#    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes, activation='sigmoid'))
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=sgd,
#                  metrics=['accuracy'])
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=sgd,
                  metrics=['binary_accuracy', 'top_k_categorical_accuracy'])
    return model

model = get_model()
model.summary()


# Predicts one sample
def predict(sample, model):
#    sample = mfcc_operation(wavfile_path = filepath,max_len = feature_dim_2)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]
	

earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=verbose, mode='auto')
model.fit(X_train, y_train_hot,batch_size=batch_size,epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid_hot))
model.save('model-8april.h5') 
model.save_weights('model-weights-8april.h5')


#
#print('Test resuts:')
#for i,sample in enumerate(X_test):
#    print('correct:', get_labels()[0][int(y_test[i])], 'prediction:', predict(sample=sample,model=model))



#for testfile in os.listdir('D:/Python projects/Speaker_Recognition/test'):
#    print('We Predict file:', testfile)
#    print(predict(filepath='D:/Python projects/Speaker_Recognition/test/' + testfile, model=model))
#
