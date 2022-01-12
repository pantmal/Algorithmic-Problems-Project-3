import sys, math, random, csv, os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers

import pickle
import os

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

#/---------------------------Chapter: Defining model functions------------------/

def encoder(input_window):
    x = Conv1D(32, 5, activation="relu", padding="same")(input_window) # 20 dims
    x = MaxPooling1D(2, padding="same")(x) # 10 dims
    
    #x = Conv1D(32, 5, activation="relu", padding="same")(x) # 10 dims
    #x = MaxPooling1D(3, padding="same")(x) # 5 dims

    x = Conv1D(1, 5, activation="relu", padding="same")(x) # 5 dims
    encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

    return encoded


def decoder(encoded):
    
    x = UpSampling1D(2)(encoded) # 6 dims
    x = Conv1D(32, 5, activation="relu", padding="same")(x) # 3 dims
    #x = UpSampling1D(3)(x) # 6 dims
    
    #x = Conv1D(32, 5, activation='relu')(x) # 5 dims
    x = UpSampling1D(2)(x) # 10 dims
    
    #x = Conv1D(32, 5, activation='relu')(x) # 20 dims

    decoded = Conv1D(1, 5, activation='sigmoid', padding='same')(x) # 10 dims
    
    return decoded


#/------------------------Chapter: Parameter setup------------------/

dataset_path = ''
queryset_path = ''
output_dataset_path = ''
output_queryset_path = ''
window_length = 20
predict = False

input_args = sys.argv
for i in range(1, len(input_args)): 
    if input_args[i] == '-d':
        dataset_path = input_args[i+1]
    if input_args[i] == '-q':
        queryset_path = input_args[i+1]
    if input_args[i] == '-od':
        output_dataset_path = input_args[i+1]
    if input_args[i] == '-oq':
        output_queryset_path = input_args[i+1]
    if input_args[i] == '-predict':
        predict = True

if dataset_path == '':
    print('You must specify a dataset')
    sys.exit()

if queryset_path == '':
    print('You must specify a query set')
    sys.exit()

if output_dataset_path == '':
    print('You must specify a reduced dataset path')
    sys.exit()

if output_queryset_path == '':
    print('You must specify a reduced query path')
    sys.exit()

#/------------------------Chapter: Train/Test split-------------------------/

df = pd.read_csv(dataset_path, '\t', header=None)
qf = pd.read_csv(queryset_path, '\t', header=None)

split_num = int((0.8 * (df.shape[1]-1)))

training_sets = []
test_sets = []
for i in range(df.shape[0]):
    training_sets.append(df.iloc[i , 1:split_num+1].values)
    test_sets.append(df.iloc[i , split_num+1: ].values)

split_num2 = int((0.8 * (qf.shape[1]-1)))

for i in range(qf.shape[0]):
    training_sets.append(qf.iloc[i , 1:split_num2+1].values)
    test_sets.append(qf.iloc[i , split_num2+1: ].values)

#/-----------------------Chapter: Defining feature arrays----------------------/

scalers = []

X_trains = []
for sett in training_sets:
    training_set_scaled = np.reshape(sett, (-1,1)) 
    training_set_scaled = np.asarray(training_set_scaled).astype('float32')

    sc = MinMaxScaler(feature_range = (0, 1))
    # Creating a data structure with 60 time-steps and 1 output
    training_set_scaled = sc.fit_transform(training_set_scaled)

    scalers.append(sc)

    X_train = []
    y_train = []
    
    for i in range(0, len(training_set_scaled), window_length):
        X_train.append(training_set_scaled[i:(i + window_length)])
    
    if len(X_train[-1]) != window_length:
        del X_train[-1]

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #print(X_train.shape)

    X_trains.append(X_train)


X_tests = []
for (sett,sc) in zip(test_sets,scalers):
    test_set_scaled = np.reshape(sett, (-1,1)) 
    test_set_scaled = np.asarray(test_set_scaled).astype('float32')

    test_set_scaled = sc.transform(test_set_scaled)
    X_test = []
    y_test = []
    for i in range(0, len(test_set_scaled), window_length):
        X_test.append(test_set_scaled[i:(i + window_length)])
    
    if len(X_test[-1]) != window_length:
        del X_test[-1]

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)

    X_tests.append(X_test)

X_train_big = [item for sublist in X_trains for item in sublist]
X_train_big = np.array(X_train_big)

X_test_big = [item for sublist in X_tests for item in sublist]

#/--------------------Chapter: Model definition and training/predicting---------/

input_window = Input(shape=(window_length,1))
encoder = Model(input_window, encoder(input_window))
autoencoder = Model(input_window, decoder(encoder(input_window)))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
autoencoder.summary()

if predict == False:

    for (train,test) in zip(X_trains,X_tests):
        history = autoencoder.fit(train, train,epochs=5,batch_size=128,shuffle=True,validation_split=0.1)   

    autoencoder.save('reduce_model.h5')
    mae_t = np.mean(history.history['val_loss'])
    print("The average loss of the validation sets is ", mae_t)

autoencoder = tf.keras.models.load_model('reduce_model.h5')

test_losses = []
for test in X_tests:
    decoded_stock = autoencoder.predict(test, verbose = 0)
    loss = autoencoder.evaluate(decoded_stock,test, verbose = 0)    
    test_losses.append(loss)

mae_t = np.mean(np.array(test_losses)) 
print("The average loss of the test sets is ", mae_t)

#/---------------------Chapter: Writing out reduced dimension files------------/

dataset_prices = []
dataset_names = []
for i in range(df.shape[0]):
    dataset_prices.append(df.iloc[i , 1:].values)
    dataset_names.append(df.iloc[i , 0])

query_prices = []
query_names = []
for i in range(qf.shape[0]):
    query_prices.append(qf.iloc[i , 1:].values)
    query_names.append(qf.iloc[i , 0])

dataset_scalers = []
X_datasets = []
for sett in dataset_prices:
    training_set_scaled = np.reshape(sett, (-1,1)) 
    training_set_scaled = np.asarray(training_set_scaled).astype('float32')

    sc = MinMaxScaler(feature_range = (0, 1))

    training_set_scaled = sc.fit_transform(training_set_scaled)
    dataset_scalers.append(sc)
    
    X_train = []
    for i in range(0, len(training_set_scaled), window_length):
        X_train.append(training_set_scaled[i:(i + window_length)])
    
    if len(X_train[-1]) != window_length:
        del X_train[-1]

    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #print(X_train.shape)

    X_datasets.append(X_train)

query_scalers = []
X_queries = []
for sett in query_prices:

    sc = MinMaxScaler(feature_range = (0, 1))

    test_set_scaled = np.reshape(sett, (-1,1)) 
    test_set_scaled = np.asarray(test_set_scaled).astype('float32')
    
    test_set_scaled = sc.fit_transform(test_set_scaled)
    query_scalers.append(sc)
    
    X_test = []
    for i in range(0, len(test_set_scaled), window_length):
        X_test.append(test_set_scaled[i:(i + window_length)])
    
    if len(X_test[-1]) != window_length:
        del X_test[-1]

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)

    X_queries.append(X_test)

columns = 0
latent_dim = 0

with open(output_dataset_path, 'w', encoding='UTF8') as f:

    for (name, dset, sc) in zip(dataset_names, X_datasets, dataset_scalers):

        reduced_stock = []

        i = 0
        encoded_stock = encoder.predict(dset)
        columns = df.shape[1]-1
        latent_dim = encoded_stock[0].shape[0]

        for iterator in range(len(encoded_stock)):
            slc = sc.inverse_transform(encoded_stock[iterator])
            for item in slc:
                reduced_stock.append(item)
    
        reduced_stock = [item for sublist in reduced_stock for item in sublist]
        
        row_str = []
        id = name
        row_str.append(id)
        for j in reduced_stock:
            row_str.append(j)
        
        writer = csv.writer(f)
        writer.writerow(row_str) 

with open(output_queryset_path, 'w', encoding='UTF8') as f:

    for (name, query,sc) in zip(query_names, X_queries, query_scalers):

        reduced_stock = []

        i = 0
        encoded_stock = encoder.predict(query)
        for iterator in range(len(encoded_stock)):
            slc = sc.inverse_transform(encoded_stock[iterator])
            
            for item in slc:
                reduced_stock.append(item)
          
        reduced_stock = [item for sublist in reduced_stock for item in sublist]
        
        row_str = []
        id = name
        row_str.append(id)
        for j in reduced_stock:
            row_str.append(j)
        
        writer = csv.writer(f)
        writer.writerow(row_str) 

print('The number of reduced columns will be: ' + str( int(columns/window_length) * latent_dim) )