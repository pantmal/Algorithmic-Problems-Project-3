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


#/------------------------Chapter: Parameter setup------------------/

n = 1
total = False
graphs_all = False
predict = False
time_steps = 50
graphs_n = 1 
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
    if input_args[i] == '-graphs_n':
        graphs_all = True
        graphs_n = int(input_args[i+1])

if graphs_all == True:
    graphs_n = n

if dataset == '':
    print('You must specify a dataset.')
    sys.exit()

if graphs_n > n:
    print('Selected graphs number is larger than time series selected.')
    sys.exit()

#/------------------------Chapter: Train/Test split-------------------------/

df = pd.read_csv(dataset, '\t', header=None)

df_rand = random.sample(range(df.shape[0]), n) 

split_num = int((0.8 * (df.shape[1]-1)))

training_sets = []
test_sets = []
input_sets = []
names = []
for i in df_rand:

    if total == False:
        training_sets.append(df.iloc[i , 1:split_num+1].values)

    names.append(df.iloc[i,0])
    test_set = df.iloc[i , split_num+1: ].values
    test_sets.append(test_set)

    input_sets.append(df.iloc[i, 0+(df.shape[1] - len(test_set) - time_steps) :].values) 


if total == True:
   for i in range(df.shape[0]):
       training_sets.append(df.iloc[i , 1:split_num+1].values)
    


#/-----------------------Chapter: Defining feature arrays----------------------/

scalers = []

X_trains = []
y_trains = []
for sett in training_sets:
    
    sc = StandardScaler()
    # Creating a data structure with 60 time-steps and 1 output
    training_set_scaled = np.reshape(sett, (-1,1)) 
    training_set_scaled = sc.fit_transform(training_set_scaled)

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
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_test.shape)
    
    X_tests.append(X_test)

X_train_big = [item for sublist in X_trains for item in sublist]
y_train_big = [item for sublist in y_trains for item in sublist]
X_train_big = np.array(X_train_big)
y_train_big = np.array(y_train_big)


#/--------------------Chapter: Model definition and training/predicting---------/
val_losses = []
test_losses = []
pred_prices = []
if total == False:

    for (x,y,test,sc, test_og) in zip(X_trains, y_trains, X_tests,scalers, test_sets):
    
        K.clear_session()
        model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_trains[0].shape[1], 1)))
        model.add(Dropout(0.7))# Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True))
        model.add(Dropout(0.7))# Adding a third LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True))
        model.add(Dropout(0.7))# Adding a fourth LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64))
        model.add(Dropout(0.7))# Adding the output layer
        model.add(Dense(units = 1))
        
        #Compiling the RNN
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        hist = model.fit(x, y, epochs = 7, batch_size = 128, validation_split=0.1)
        print()
    
        val_losses.append(np.mean(hist.history['val_loss'])) 

        pred_price = model.predict(test)
        pred_price = sc.inverse_transform(pred_price)
        test_mae_loss = np.mean(np.abs(pred_price - test_og), axis=1)
        test_losses.append(test_mae_loss)

        pred_prices.append(pred_price)
    
else:

    if predict == False:
        model = Sequential()#Adding the first LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_big.shape[1], 1)))
        model.add(Dropout(0.5))# Adding a second LSTM layer and some Dropout regularisation
        model.add(LSTM(units = 64))
        model.add(Dropout(0.5))# Adding the output layer
        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        hist = model.fit(X_train_big, y_train_big, epochs = 5, batch_size = 512, validation_split=0.1)    
        
        model.save('forecast_model.h5')

        val_losses.append(np.mean(hist.history['val_loss']))
    
    model = tf.keras.models.load_model('forecast_model.h5')

    for (test,sc,test_og) in zip(X_tests,scalers,test_sets):
        pred_price = model.predict(test)
        pred_price = sc.inverse_transform(pred_price)
        test_mae_loss = np.mean(np.abs(pred_price - test_og), axis=1)
        test_losses.append(test_mae_loss)

        pred_prices.append(pred_price)
    
if predict == False:
    print("The average loss of the validation sets is ", np.mean(val_losses))

print("The average loss of the test sets is ", np.mean(test_losses))

#/------------------------------Chapter: Graphs-------------------------------/

# Visualising the results

random_stock = random.choice(range(n))

if graphs_all == False:
    name = names[random_stock]
    plt.figure()
    plt.plot(range(split_num,df.shape[1]-1),test_sets[random_stock], color = 'red', label = 'True ' + name)
    plt.plot(range(split_num,df.shape[1]-1),pred_prices[random_stock], color = 'blue', label = 'Predicted ' + name)
    plt.xticks(np.arange(split_num,df.shape[1]-1,time_steps*2))
    plt.title(name + ' Prediction')
    plt.xlabel('Time values')
    plt.ylabel(name)
    plt.legend()
    plt.show()
else:
    for stock in range(n):
        if stock == graphs_n:
          break

        name = names[stock]
        plt.figure()    
        plt.plot(range(split_num,df.shape[1]-1),test_sets[stock], color = 'red', label = 'True ' + name)
        plt.plot(range(split_num,df.shape[1]-1),pred_prices[stock], color = 'blue', label = 'Predicted ' + name)
        plt.xticks(np.arange(split_num,df.shape[1]-1,time_steps*2))
        plt.title(name + ' Prediction')
        plt.xlabel('Time values')
        plt.ylabel(name)
        plt.legend()
        plt.show()
