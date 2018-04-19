# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:34:00 2018

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
from keras.models import Model
from keras import regularizers
import numpy as np
from keras.callbacks import EarlyStopping
from keras import metrics

feature_dim_1 = 40
feature_dim_2 = 40
channel = 1

verbose = 1

num_classes = 2

epochs = 40
batch_size = 5

#save_random_user_data()
#save_user_data_as_numpy(max_len=feature_dim_1,max_len2=feature_dim_2)

X_train, X_valid, y_train, y_valid= get_train_test(split_ratio=0.8, random_state=42,maxsamples=40,path=enrolldata)
#mean=  -2.0341127496466784
#std=  13.643874623449355

std = np.std(X_train)
mean = np.mean(X_train)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_train = X_train - mean
X_train = X_train / std
print('after sub mean and div by std')
print('X train mean',np.mean(X_train))
print('X train std',np.std(X_train))

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


#model = load_model('model-8april.h5')
#model.load_weights('model-weights-8april.h5')
#model.layers.pop()
#model.add(Dense(num_classes, activation='sigmoid', name='addedDense'))


#Loading pre-trained model
tempmodel = load_model('model-8april.h5')
tempmodel.summary()
new_layer = Dense(2, activation='sigmoid', name='my_dense')
inp = tempmodel.input
#using pre trained model except last 2 layers
out = new_layer(tempmodel.layers[-2].output)

#Creating new model
model = Model(inp, out)
model.summary()

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adamax',
                  metrics=['binary_accuracy'])
model.fit(X_train, y_train_hot,batch_size=batch_size,epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid_hot))


model.save('tester-model.h5')

def predict(sample, model):
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels(enrolldata)[0][np.argmax(model.predict(sample_reshaped))]

false = 0
true = 0

print('Test resuts:')
for i,sample in enumerate(X_test):
    print('correct:', get_labels(enrolldata)[0][int(y_test[i])], 'prediction:', predict(sample=sample,model=model))
    if get_labels(enrolldata)[0][int(y_test[i])] != predict(sample=sample,model=model):
        false = false + 1
    else: true = true + 1

acc = true / X_test.shape[0]
print(acc)




