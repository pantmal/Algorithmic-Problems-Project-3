import sys, math, random, os
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

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go

#/------------------------Chapter: Parameter setup------------------/

n = 5
mae = 0.09
time_steps = 50 #TODO
graphs_all = False
predict = False
loss_plot = False
graphs_n = 5#TODO
dataset = ''

input_args = sys.argv
for i in range(1, len(input_args)): 
    if input_args[i] == '-d':
        dataset = input_args[i+1]
    if input_args[i] == '-n':
        n = int(input_args[i+1])
    if input_args[i] == '-graphs_all':
        graphs_all = True
    if input_args[i] == '-loss_plot':
        loss_plot = True
    if input_args[i] == '-predict':
        predict = True
    if input_args[i] == '-mae':
        mae = float(input_args[i+1])
    if input_args[i] == '-lookback':
        time_steps = int(input_args[i+1])
    if input_args[i] == '-graphs_n':
        graphs_all = True
        graphs_n = int(input_args[i+1])

if dataset == '':
    print('You must specify a dataset')
    sys.exit()

if graphs_n > n:
    print('Selected graphs number is larger than time series selected.')
    sys.exit()

#/------------------------Chapter: Train/Test split-------------------------/
df = pd.read_csv(dataset, '\t', header=None)

#print("Number of rows and columns:", df.shape)

#df_rand = random.sample(range(df.shape[0]), n) 

# split_num = int((0.8 * (df.shape[1]-1)))
# for i in df_rand:
#     training_sets.append(df.iloc[i , 1:split_num+1].values)
#     names.append(df.iloc[i,0])
#     test_set = df.iloc[i , split_num+1: ].values
#     test_sets.append(test_set)

split_num = int((0.8 * (df.shape[0]-1)))
t_num = df.shape[0] - split_num

names = []
training_sets = []
test_sets = []

for i in range(split_num):
    training_sets.append(df.iloc[i , 1:].values)
 
    
for i in range(t_num):
    names.append(df.iloc[i,0])
    test_set = df.iloc[i , 1: ].values
    test_sets.append(test_set)

if n > len(test_sets):
    print('Selected time series number is larger than the length of the test set ('+ str(len(test_sets))+').')
    print('Will use n = length of test set, in order to proceed.')
    n = len(test_sets)
    graphs_n = n
    


#training_set_big = [item for sublist in training_sets for item in sublist]
#test_set_big = [item for sublist in test_sets for item in sublist]

#/-----------------------Chapter: Defining feature arrays----------------------/

scalers = []

X_trains = []
y_trains = []
for sett in training_sets:
    training_set_scaled = np.reshape(sett, (-1,1)) 

    sc = StandardScaler()
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
for (sett,sc) in zip(test_sets,scalers):
    test_set_scaled = np.reshape(sett, (-1,1)) 

    test_set_scaled = sc.transform(test_set_scaled)# Creating a data structure with 60 time-steps and 1 output
    X_test = []
    y_test = []
    for i in range(len(test_set_scaled)-time_steps):
        X_test.append(test_set_scaled[i:(i+time_steps)])
        y_test.append(test_set_scaled[i+time_steps])

    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(X_train.shape)
    X_tests.append(X_test)

X_train_big = [item for sublist in X_trains for item in sublist]
y_train_big = [item for sublist in y_trains for item in sublist]
X_train_big = np.array(X_train_big)
y_train_big = np.array(y_train_big)
#print(X_train_fit.shape)


#/--------------------Chapter: Model definition and training/predicting---------/

val_losses = []
if predict == False:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train_big.shape[1],1)))
    model.add(Dropout(rate=0.2))
    #model.add(RepeatVector(X_trains[0].shape[1]))
    #model.add(LSTM(64, return_sequences=True))
    #model.add(Dropout(rate=0.2))
    #model.add(LSTM(64, return_sequences=True))
    #model.add(Dropout(rate=0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    #Compiling the model
    #for (x,y) in zip(X_train_fits,y_train_fits): 
    hist = model.fit(X_train_big, y_train_big, epochs=5, batch_size=512, validation_split=0.1, shuffle=False)
    val_losses.append(np.mean(hist.history['val_loss']))
    model.save('detect_model.h5')

print("The average loss of the validation sets is ", np.mean(val_losses))

# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()

#model.evaluate(X_test, y_test)

#/-------------------------------Chapter: Getting losses----------------------/

model = tf.keras.models.load_model('detect_model.h5')

# thresholds = []
# for train in X_trains:
#     X_train_pred = model.predict(train, verbose=0)
#     train_mae_loss = np.mean(np.abs(X_train_pred - train), axis=1)

#     threshold = np.max(train_mae_loss)
#     thresholds.append(threshold)
    #print(f'Reconstruction error threshold: {threshold}')

#mae_t = np.mean(np.array(thresholds)) 
#print(mae_t)

#plt.hist(train_mae_loss, bins=50)
#plt.xlabel('Train MAE loss')
#plt.ylabel('Number of Samples');

#plt.show()
#TODO: ADD N HERE
test_losses = []
for test in X_tests:
    X_test_pred = model.predict(test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred - test), axis=1)

    test_losses.append(test_mae_loss)
    #print(f'Reconstruction error threshold: {test_mae_loss}')

mean_test = np.mean(np.array(test_losses)) 
print("The average loss of the test sets is ", mean_test)

#plt.hist(test_mae_loss, bins=50)
#plt.xlabel('Test MAE loss')
#plt.ylabel('Number of samples')
#threshold2 = np.max(test_mae_loss)
#plt.show()


#/---------------------------Chapter: Graphs-----------------------------------/

random_stock = random.choice(range(n)) 

threshold = mae

if graphs_all == False:
    
    test_score_df = pd.DataFrame(test_sets[random_stock][time_steps:])
    test_score_df['loss'] = test_losses[random_stock]
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['Close'] = test_sets[random_stock][time_steps:]

    #x_range = (df.shape[1]-1)-split_num - time_steps
    x_range = (df.shape[1]-1) - time_steps

    x = []
    for i in range(x_range):
        x.append(i)

    test_score_df['date'] = x

    if loss_plot:

        plt.subplot(2, 1, 1)
        plt.plot(x,test_score_df['loss'], color = 'blue', label = 'Test loss')
        plt.plot(x,test_score_df['threshold'], color = 'red', label = 'Threshold')
        #plt.xticks(np.arange(0,x_range,time_steps))
        plt.title('Test loss vs. Threshold')
        #plt.xlabel('Time')
        #plt.ylabel('TESLA Stock Price')
        plt.legend()

        plt.subplot(2, 1, 2)

    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    #print(anomalies.shape)

    anomalies_x = []
    for i in range(anomalies.shape[0]):
        anomalies_x.append(i)

    plt.plot(x,test_score_df['Close'], color = 'blue', label = 'Close price ' + names[random_stock])
    plt.plot(anomalies['date'],anomalies['Close'], color = 'red', marker = '.', label = 'Anomalies',linestyle='None')
    plt.xticks(np.arange(0,x_range,time_steps*10))
    plt.title('Anomalies detected')
    plt.xlabel('Time')
    #plt.ylabel('TESLA Stock Price')
    plt.legend()
    plt.show()

else:

    for stock in range(n):
        if stock == graphs_n:
          break


        test_score_df = pd.DataFrame(test_sets[stock][time_steps:])
        test_score_df['loss'] = test_losses[stock]
        test_score_df['threshold'] = threshold
        test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
        test_score_df['Close'] = test_sets[stock][time_steps:]

        #x_range = (df.shape[1]-1)-split_num - time_steps
        x_range = (df.shape[1]-1) - time_steps

        x = []
        for i in range(x_range):
            x.append(i)

        test_score_df['date'] = x

        if loss_plot:

            plt.subplot(2, 1, 1)
            plt.plot(x,test_score_df['loss'], color = 'blue', label = 'Test loss')
            plt.plot(x,test_score_df['threshold'], color = 'red', label = 'Threshold')
            #plt.xticks(np.arange(0,x_range,time_steps))
            plt.title('Test loss vs. Threshold')
            #plt.xlabel('Time')
            #plt.ylabel('TESLA Stock Price')
            plt.legend()

            plt.subplot(2, 1, 2)

        anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
        #print(anomalies.shape)

        anomalies_x = []
        for i in range(anomalies.shape[0]):
            anomalies_x.append(i)

        plt.plot(x,test_score_df['Close'], color = 'blue', label = 'Close price ' + names[stock])
        plt.plot(anomalies['date'],anomalies['Close'], color = 'red', marker = '.', label = 'Anomalies',linestyle='None')
        plt.xticks(np.arange(0,x_range,time_steps*10))
        plt.title('Anomalies detected')
        plt.xlabel('Time')
        #plt.ylabel('TESLA Stock Price')
        plt.legend()
        plt.show()
