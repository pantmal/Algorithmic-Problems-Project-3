import sys, math, random, os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pandas as pd
import numpy as np

from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler


#CHAPTER: PARAMS

n = 5
total = False
graphs_all = False
predict = False
time_steps = 10
dataset = ''

input_args = sys.argv
for i in range(1, len(input_args)):
    if input_args[i] == '-d':
        dataset = input_args[i+1]
    if input_args[i] == '-n':
        n = int(input_args[i+1])
    if input_args[i] == '-total':
        total = True
    if input_args[i] == '-graphs_all':
        graphs_all = True
    if input_args[i] == '-predict':
        predict = True
    if input_args[i] == '-lookback':
        time_steps = int(input_args[i+1])
    
if dataset == '':
    print('You must specify a dataset')
    sys.exit()

#CHAPTER: SPLITS
df = pd.read_csv(dataset, '\t', header=None)

print("Number of rows and columns:", df.shape)

df_rand = random.sample(range(df.shape[0]), n) 

split_num = int((0.8 * (df.shape[1]-1)))
#print(split_num)
training_sets = []
test_sets = []
input_sets = []
names = []
for i in df_rand:
    
    #i = 7

    #if total == False:
    training_sets.append(df.iloc[i , 1:split_num+1].values)

    names.append(df.iloc[i,0])
    test_set = df.iloc[i , split_num+1: ].values
    test_sets.append(test_set)

    input_sets.append(df.iloc[i, 0+(df.shape[1] - len(test_set) - time_steps) :].values) 
    #print(len(input_sets[i]))

#if total == True:
#    for i in range(df.shape[0]):
#        training_sets.append(df.iloc[i , 1:split_num+1].values)
    

#print(df.iloc[len(dataset_total) - len(dataset_test) - time_steps + 1])
#print(df.iloc[0, 1+(len(dataset_total) - len(dataset_test) - time_steps) : ].values)
#print(df.iloc[0, 1+(len(dataset_total) - len(dataset_test) - time_steps) : ].shape)
#print(df.shape[1])    

training_set_big = [item for sublist in training_sets for item in sublist]
test_set_big = [item for sublist in test_sets for item in sublist]

#CHAPTER: FEATURES

scalers = []

# Feature Scaling
X_trains = []
y_trains = []
for sett in training_sets:
    
    sc = StandardScaler()

    training_set_scaled = np.reshape(sett, (-1,1)) 
    training_set_scaled = sc.fit_transform(training_set_scaled)# Creating a data structure with 60 time-steps and 1 output

    scalers.append(sc)

    X_train = []
    y_train = []
    for i in range(len(training_set_scaled)-time_steps):
        X_train.append(training_set_scaled[i:(i+time_steps)])
        y_train.append(training_set_scaled[i+time_steps])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    #print(X_train.shape)
    X_trains.append(X_train)
    y_trains.append(y_train)


X_tests = []
y_tests = []
for (sett, sc) in zip(input_sets,scalers):
    
    inputs = sett.reshape(-1,1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(len(inputs) - time_steps):
        X_test.append(inputs[i:(i+time_steps)])
        #X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)
    X_tests.append(X_test)

X_train_big = [item for sublist in X_trains for item in sublist]
y_train_big = [item for sublist in y_trains for item in sublist]
X_train_big = np.array(X_train_big)
y_train_big = np.array(y_train_big)


#CHAPTER: FIT
test_losses = []
pred_prices = []
if total == False:

    for (x,y,test,sc) in zip(X_trains, y_trains, X_tests,scalers):
    
        K.clear_session()
        model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_trains[0].shape[1], 1)))
        model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))# Adding the output layer
        model.add(Dense(units = 1))
        
        # Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        hist = model.fit(x, y, epochs = 10, batch_size = 64, validation_split=0.1)
        #model.save('per_one.h5')
        print()
    
        #model = tf.keras.models.load_model('per_one.h5')
        test_losses.append(np.mean(hist.history['val_loss'])) 

        pred_price = model.predict(test)
        pred_price = sc.inverse_transform(pred_price)
        pred_prices.append(pred_price)
    
else:

    if predict == False:
        model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_big.shape[1], 1)))
        model.add(Dropout(0.2))# Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))# Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))# Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))# Adding the output layer
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        hist = model.fit(X_train_big, y_train_big, epochs = 15, batch_size = 64, validation_split=0.1)    
        
        model.save('forecast_model.h5')

        test_losses.append(np.mean(hist.history['val_loss']))
    
    model = tf.keras.models.load_model('forecast_model.h5')

    for (test,sc) in zip(X_tests,scalers):
        pred_price = model.predict(test)
        pred_price = sc.inverse_transform(pred_price)
        pred_prices.append(pred_price)
    

print(np.mean(test_losses))
#print(len(pred_prices))
#print(X_test.shape)
# (459, 60, 1)

#predicted_stock_price = model.predict(X_test)
#predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#CHAPTER: GRAPH

# Visualising the results

random_stock = random.choice(range(n))

if graphs_all == False:
    name = names[random_stock]
    plt.figure()    
    plt.plot(range(split_num,df.shape[1]-1),test_sets[random_stock], color = 'red', label = name)
    plt.plot(range(split_num,df.shape[1]-1),pred_prices[random_stock], color = 'blue', label = 'Predicted ' + name)
    plt.xticks(np.arange(split_num,df.shape[1]-1,time_steps*10))
    plt.title(name + ' Prediction')
    plt.xlabel('Time')
    plt.ylabel(name)
    plt.legend()
    plt.show()
else:
    for stock in range(n):
        name = names[stock]
        plt.figure()    
        plt.plot(range(split_num,df.shape[1]-1),test_sets[stock], color = 'red', label = name)
        plt.plot(range(split_num,df.shape[1]-1),pred_prices[stock], color = 'blue', label = 'Predicted ' + name)
        plt.xticks(np.arange(split_num,df.shape[1]-1,time_steps*10))
        plt.title(name + ' Prediction')
        plt.xlabel('Time')
        plt.ylabel(name)
        plt.legend()
        plt.show()
