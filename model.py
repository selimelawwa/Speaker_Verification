# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:42:37 2018

@author: selem
"""

from preprocess import *
import keras  
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import load_model



feature_dim_1 = 48
feature_dim_2 = 48
channel = 1
epochs = 50
batch_size = 100
num_classes = 40
verbose = 1

save_data_as_numpy_array(max_len = feature_dim_2)


X_train, X_valid, y_train, y_valid = get_train_test()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True)

# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_valid = X_valid.reshape(X_valid.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
#one hot encoding of Y
y_train_hot = to_categorical(y_train)
y_valid_hot = to_categorical(y_valid)
y_test_hot = to_categorical(y_test)



def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

#def get_model():
#    model = Sequential()
#    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Flatten())
#    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(Dense(256, activation='relu'))
#    model.add(Dropout(0.25))
#    model.add(Dense(num_classes, activation='softmax'))
#    model.compile(loss=keras.losses.categorical_crossentropy,
#                  optimizer=keras.optimizers.Adadelta(),
#                  metrics=['accuracy'])
#    return model

# Predicts one sample
def predict(sample, model):
#    sample = mfcc_operation(wavfile_path = filepath,max_len = feature_dim_2)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][np.argmax(model.predict(sample_reshaped))]
	
#model = get_model()
#model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid_hot))
#model.save('my_model.h5') 

model = load_model('my_model.h5') 

print('Test resuts:')
for i,sample in enumerate(X_test):
    print('correct:', get_labels()[0][int(y_test[i])], 'prediction:', predict(sample=sample,model=model))



#for testfile in os.listdir('D:/Python projects/Speaker_Recognition/test'):
#    print('We Predict file:', testfile)
#    print(predict(filepath='D:/Python projects/Speaker_Recognition/test/' + testfile, model=model))
#
