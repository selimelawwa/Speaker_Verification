# -*- coding: utf-8 -*-
"""
Created on Tue May  8 22:58:20 2018

@author: selem
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:34:00 2018

@author: selem
"""
from preprocess import *
from enrollment_helpers import *
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


#Loading pre-trained model
loaded_model = load_model('firstbigmodel.h5')
#data specs
feature_dim_1 = loaded_model.input.shape[1]  #time
feature_dim_2 = loaded_model.input.shape[2]  #frequency
channel = 1

#you have to make sure of model name, numpy source path

verbose = 1
num_classes = 2
epochs = 5
batch_size = 5

save_random_user_data(origin_path=big_numpy_files_path)
save_user_data_as_numpy(max_len=feature_dim_1,max_len2=feature_dim_2)

X_train, X_valid, X_test ,y_train,y_valid,y_test = get_x_y_data(split_ratio=0.8, random_state=42,maxsamples=30,path=enroll_data_path,mode=None)
X_train, X_valid, X_test = reshape_prepare_for_input(X1=X_train, X2=X_valid, X3=X_test,channel=channel)
#one hot encoding of Y

y_train_hot = to_categorical(y_train)
y_valid_hot = to_categorical(y_valid)
y_test_hot = to_categorical(y_test)




inp = loaded_model.input
#x = Dense(128, activation='relu',name='added_dense_1')(loaded_model.layers[-2].output)
#x2 = Dense(64, activation='relu',name='added_dense_2')(x)
#out = Dense(1, activation='sigmoid', name='my_dense')(x2)
out = Dense(1, activation='sigmoid', name='my_dense')(loaded_model.layers[-2].output)
#Creating new model
model = Model(inp, out)
model.summary()
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adamax',
                  metrics=['binary_accuracy'])
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=15, verbose=verbose, mode='auto')

#model.fit(X_train, y_train_hot,batch_size=batch_size,epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid_hot))
model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs, verbose=verbose, validation_data=(X_valid, y_valid))


#generate random test data
save_data_as_numpy_array(max_len = feature_dim_1, max_len2 = feature_dim_2,origin_path=test_data_path,destination_path=numpy_test_path)
new_x,new_y = get_x_y_data(maxsamples=90,path=numpy_test_path,mode=1)

#y=1 user, y=0 not user
labels = get_labels(numpy_test_path)
print('labels[0] is for:', labels[0])
print('labels[1] is for:', labels[1])
def test(x1,y1):
    acc=0
    true=0
    false_acceptnce=0
    false_rejection=0
    for i,sample in enumerate(x1):
        sample = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
        pred = model.predict(sample)
        correct = int(y1[i])
        if pred>0.9:
            result= 1
        else: result = 0
        
        if correct==result:
            true+=1
            print('CORRECT --- model output: ',pred,' correct: ', labels[correct],' Result: ',labels[result])
        else:
            if(correct==0):
                false_acceptnce+=1
            else:
                false_rejection+=1
            print('FALSE --- model output: ',pred,' correct: ', labels[correct],' Result: ',labels[result])
    acc = true / x1.shape[0]
    print('Testing with ',x1.shape[0],'samples got an accuracy: ',acc)
    print('False acceptance: ',false_acceptnce)
    print('False rejection: ',false_rejection)
test(new_x,new_y)
