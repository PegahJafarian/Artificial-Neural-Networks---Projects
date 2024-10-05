import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import random
from math import sqrt
from collections import deque

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics                               # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Activation, RepeatVector, CuDNNGRU, LSTMCell
from tensorflow.keras.layers import Flatten, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import random
from math import sqrt
from collections import deque
from sklearn import metrics                               
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics                               
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Activation, Bidirectional
from tensorflow.keras.layers import Flatten, TimeDistributed

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


SEQ_LEN = 100                    
FUTURE_PERIOD_PREDICT = 6        
EPOCHS = 30                      
BATCH_SIZE = 64                  




def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def theil_u(yhat, y):
  
  len_yhat = len(yhat)
  sum1 = 0
  sum2 = 0
  
  for i in range(len_yhat-1):
    
    sum1 += ( (yhat[i+1] - y[i+1]) / y[i] )**2
    sum2 += ( (y[i+1] - y[i]) / y[i] )**2
    
  return sqrt(sum1/sum2)


def plot_history(loss, val_loss, epochs):
  
  plt.figure(figsize=[15,8])
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(range(epochs), loss, label='Train Error')
  plt.plot(range(epochs), val_loss, label = 'Val Error')
  plt.ylim([0.05,0.15])
  plt.legend()
  plt.show()


def preprocess_df(df, shuffle=False):
  
    sequential_data = []                    
    prev_days = deque(maxlen = SEQ_LEN)    
                                          
                                           
    for i in df.values:                                             
        prev_days.append([n for n in i[:-1]])                      
        if len(prev_days) == SEQ_LEN:                            
            sequential_data.append([np.array(prev_days), i[-1]])    
            
            
    if(shuffle == True):
      random.shuffle(sequential_data)                               
    

    X = []
    y = []

    for seq, target in sequential_data:  
        X.append(seq)                   
        y.append(target)                

    return np.array(X), np.array(y)  
  
  
def train_length(length, batch_size):
  
  length_values = []
  for x in range( int(length)-100, int(length) ):
    modulo = x % batch_size
    if(modulo == 0 ):
      length_values.append(x)
  
  return ( max(length_values) )


def split_data(df, percent):
  
  length = len(df)
  test_index = int(length*percent)   
  train_index = length - test_index
  test_df = df[train_index:]                       
  df = df[:train_index] 
  
  return test_df, df


def get_model():
  
  model = Sequential()

  model.add(CuDNNLSTM(36, stateful=True, return_sequences=True, 
                 batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2])))
  model.add(Activation('relu'))
  
  model.add(CuDNNLSTM(36, stateful=True, return_sequences=False))
  model.add(Activation('relu'))
  
  model.add(Dense(1))
                  

  
  model.compile(loss = 'mae',
              optimizer = 'adam',
              metrics = ['mse', 'mae'])
  
  return model


df = pd.read_csv('kaggle_data_1h.csv', sep=',', infer_datetime_format=True, low_memory=False, index_col='time', encoding='utf-8')



df = df.drop(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)


df.index = pd.to_datetime(df.index)
df.sort_index(inplace = True)
df.dropna(inplace=True)


df = df.resample('1h').mean()
df.dropna(inplace=True)


print(df.head())


df = df.ewm(alpha=0.2).mean()
df.dropna(inplace=True)


factor = 3
upper_lim = df['Global_active_power'].mean () + df['Global_active_power'].std () * factor
lower_lim = df['Global_active_power'].mean () - df['Global_active_power'].std () * factor
df = df[(df['Global_active_power'] < upper_lim) & (df['Global_active_power'] > lower_lim)]


times = pd.to_datetime(df.index)
times = times.strftime("%Y-%m-%d").tolist()
temp = []
for index,value in enumerate(times):
  if(value > '2007-08-08' and value <'2007-09-01'):
     
    temp.append(float(df['Global_active_power'][index]))
    
temp = pd.Series(temp)
temp = temp.ewm(alpha=0.15).mean()
temp = temp.values

k=0
for index,value in enumerate(times):
  if(value > '2008-08-08' and value <'2008-09-01'):
    df['Global_active_power'].iloc[index] = temp[k]
    k += 1

print(f'df_shape arxika: {df.shape}\n')



df['future'] = df["Global_active_power"].shift(-FUTURE_PERIOD_PREDICT)
df.dropna(inplace=True)

print(df[['Global_active_power','future']].head(10))



count = 0


evaluate_loss = []
evaluate_mse = []
evaluate_mlog = []
evaluate_rmse = []
evaluate_r2 = []
evaluate_mape = []


val_loss = []
loss = []


tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(df):

 
  if (count >= 2 ):

    df2 = df.copy()

    print(f'df:{df2[:len(train_index)].shape}  test_df:{df2[len(train_index):len(train_index)+len(test_index)].shape}')

    test_df = df2[len(train_index):len(train_index)+len(test_index)]
    df2 = df2[:len(train_index)]
    valid_df, df2 = split_data(df2, 0.1)

    print(f'df_shape{df2.shape}')
    print(f'test_shape{test_df.shape}')
    print(f'Valid_shape{valid_df.shape}')

    df2.dropna(inplace=True)
    valid_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

 

    scaler = MinMaxScaler(feature_range=(0, 1))

    cols = [col for col in df.columns if col not in ['time']]

    df2[cols] = scaler.fit_transform(df2[cols])
    df2.dropna(inplace=True)                                      

    test_df[cols] = scaler.transform(test_df[cols])
    test_df.dropna(inplace=True) 

    valid_df[cols] = scaler.transform(valid_df[cols])
    valid_df.dropna(inplace=True) 


    train_x, train_y = preprocess_df(df2, shuffle=True)
    valid_x, valid_y = preprocess_df(valid_df, shuffle=False)
    test_x, test_y = preprocess_df(test_df, shuffle=False)

    train_len = train_x.shape[0]
    valid_len = valid_x.shape[0]
    test_len = test_x.shape[0]
    print(f'train_len before:{train_len} test_len:{test_len}')



    train_len = train_length(train_len, BATCH_SIZE)
    valid_len = train_length(valid_len, BATCH_SIZE)
    test_len = train_length(test_len, BATCH_SIZE)

    print(f'train_len after modulo:{train_len} test_len:{test_len}')

    train_x = train_x[:train_len]
    train_y = train_y[:train_len]

    valid_x = valid_x[:valid_len]
    valid_y = valid_y[:valid_len]

    test_x = test_x[:test_len]
    test_y = test_y[:test_len]

    print(f'Train_x shape:{train_x.shape} Train_y shape:{train_y.shape}')
    print(f'valid_x shape:{valid_x.shape} valid_y shape:{valid_y.shape}')
    print(f'Test_x shape:{test_x.shape} Test_y shape:{test_y.shape} \n')


    model = get_model()
    model.summary()


 

    for i in range(EPOCHS):
      print(f'Split: {count} Epochs: {i}/{EPOCHS}')

      history = model.fit(
        train_x, train_y,
        batch_size = BATCH_SIZE,
        epochs = epochs,
        validation_data = (valid_x, valid_y),
      )
      
      
      val_loss.append(history.history['val_loss'])
      loss.append(history.history['loss'])
      model.reset_states()

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test MSE:', score[1])
    print('Test MAE:', score[2])

    evaluate_loss.append(score[0])
    evaluate_mse.append(score[1])
    
    

    plot_history(loss, val_loss, EPOCHS) 



    yhat = model.predict(test_x)                    
                                 


    predict = np.zeros(shape=(len(yhat), 4) )       
    predict[:,0] = yhat[:,0]                        

    inv_yhat = scaler.inverse_transform(predict)   
    inv_yhat = inv_yhat[:,0]                       


       
    actual = np.zeros(shape=(len(test_y), 4) )     
    actual[:,0] = test_y

    inv_y = scaler.inverse_transform(actual)
    inv_y = inv_y[:,0]


    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse}')
    evaluate_rmse.append(rmse)

    r2 = r2_score(inv_y, inv_yhat)
    print(f'Test R^2 score: {r2}')
    evaluate_r2.append(r2)

    
  count += 1



avg_loss = np.average(evaluate_loss)
avg_mse = np.average(evaluate_mse)
avg_rmse = np.average(evaluate_rmse)
avg_r2 = np.average(evaluate_r2)
avg_theil = np.average(evaluate_theil)


print('\n')
print(f'Avg Loss:{avg_loss}')
print(f'Avg MSE:{avg_mse}')
print(f'Avg RMSE:{avg_rmse}')
print(f'Avg R2:{avg_r2}')




model.save('my_model.h5')


new_model = keras.models.load_model('my_model.h5')


keras.experimental.export_saved_model(new_model, 'SavedModel')


def mean_absolute_percentage_error(y_true, y_pred): 
   
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def theil_u(yhat, y):
  
  len_yhat = len(yhat)
  sum1 = 0
  sum2 = 0
  
  for i in range(len_yhat-1):
    
    sum1 += ( (yhat[i+1] - y[i+1]) / y[i] )**2
    sum2 += ( (y[i+1] - y[i]) / y[i] )**2
    
  return sqrt(sum1/sum2)



vals = test_df.values.copy()
cols = [col for col in df.columns if col not in ['time']]
vals = scaler.inverse_transform(vals) 
print(f'vals: {vals.shape}')
print(test_x.shape)
print(test_y.shape)



test_x1 = test_x 
test_y1 = test_y
print(test_x1.shape)
print(test_y1.shape)


yhat1 = model.predict(test_x1)  
                       




predict1 = np.zeros(shape=(len(yhat1), 4) )      
predict1[:,0] = yhat1[:,0]                         
inv_yhat1 = scaler.inverse_transform(predict1)   
inv_yhat1 = inv_yhat1[:,0]                        

actual1 = np.zeros(shape=(len(test_y1), 4) )     
actual1[:,0] = test_y1
inv_y1 = scaler.inverse_transform(actual1)
inv_y1 = inv_y1[:,0]


rmse1 = np.sqrt(mean_squared_error(inv_y1, inv_yhat1))
print(f'Test RMSE: {round(rmse1,4)}')


mae = mean_absolute_error(inv_y1, inv_yhat1)
print(f'Test MAE: {round(mae,4)}')


r21 = r2_score(inv_y1, inv_yhat1)
print(f'Test R^2 score: {round(r21,4)}')



mape = mean_absolute_percentage_error(inv_y, inv_yhat)
print(f'Test MAPE: {round(mape,4)} %\n')


plt.figure(figsize=(20,7))

plt.plot(inv_y1[500:900], color = "blue", label = 'true')
plt.plot(inv_yhat1[500:900], color = "green", label = 'prediction')
plt.xlabel('samples')
plt.ylabel('KWh')

plt.legend()
plt.show()

#Seq2Seq




keras.backend.clear_session()

layers = [30, 30]    

learning_rate = 0.001
decay = 0            


optimiser = keras.optimizers.Adam() 
num_input_features = 1              
num_output_features = 1            


loss = "mse"                      


lambda_regulariser = 0.000001                  
regulariser = None                             

batch_size = 80
epochs = 50
input_sequence_length = 72                     
target_sequence_length = 72                
num_steps_to_predict = 12                      



def plot_history(history):
  
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure(figsize=(17, 8))
  plt.xlabel('Epoch')
  plt.ylabel('mean_squared_error')
  
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'], label = 'Val Error')
  plt.ylim([0, 0.05])
  plt.legend()
  plt.show()

def preprocess_df(df, input_sequence_length, target_sequence_length, shuffle=False):
  
    SEQ_LEN = input_sequence_length + target_sequence_length
    
    sequential_data = []                    
    prev_days = deque(maxlen = SEQ_LEN)     
                                           
                                           
    for i in df:                                             
        prev_days.append(i)                                         
        if len(prev_days) == SEQ_LEN:                              
            sequential_data.append(np.array(prev_days))            
            
            
    if(shuffle == True):
      random.shuffle(sequential_data)                               
    
    X = []
    for seq in sequential_data:  
        X.append(seq)     
    
    X = np.array(X)
    
    encoder_input = X[:, :input_sequence_length, :]
    decoder_output = X[:, input_sequence_length:, :]
            
   
    decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
        
    return encoder_input, decoder_input, decoder_output

def plot_prediction(x, y_true, y_pred):

    plt.figure(figsize=(15, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j] 
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j==0 else "_nolegend_"
        label2 = "True future values" if j==0 else "_nolegend_"
        label3 = "Predictions" if j==0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1)
        plt.plot(range(len(past),
                 len(true)+len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()


def train_length(length, batch_size):
  
  length_values = []
  for x in range( int(length)-100, int(length) ):
    modulo = x % batch_size
    if(modulo == 0 ):
      length_values.append(x)
  
  return ( max(length_values) )

def split_data(df, percent):
  
  length = len(df)
  test_index = int(length*percent)   
  train_index = length - test_index
  test_df = df[train_index:]                       
  df = df[:train_index] 
  
  return test_df , df

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
   
    y_predicted = []


    states = encoder_predict_model.predict(x)


    if not isinstance(states, list):
        states = [states]


    decoder_input = np.zeros((x.shape[0], 1, 1))

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict([decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)



encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser
                                             ))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)


encoder_states = encoder_outputs_and_states[1:]



decoder_inputs = keras.layers.Input(shape=(None, 1))
decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser
                                             ))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)


decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)


decoder_outputs = decoder_outputs_and_states[0]


decoder_dense = keras.layers.Dense(num_output_features,
#                                    activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)

  
df = pd.read_csv('kaggle_data_1h.csv', sep=',', infer_datetime_format=True, low_memory=False, index_col='time', encoding='utf-8')



df = df.drop(['Voltage', 'Global_intensity', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)
df = df.round(5)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df.dropna(inplace=True)

df = df.ewm(alpha=0.15).mean()
df.dropna(inplace=True)


factor = 3
upper_lim = df['Global_active_power'].mean () + df['Global_active_power'].std () * factor
lower_lim = df['Global_active_power'].mean () - df['Global_active_power'].std () * factor
df = df[(df['Global_active_power'] < upper_lim) & (df['Global_active_power'] > lower_lim)]


times = pd.to_datetime(df.index)
times = times.strftime("%Y-%m-%d").tolist()
temp = []
for index,value in enumerate(times):
  if(value > '2007-08-08' and value <'2007-09-01'):
     
    temp.append(float(df['Global_active_power'][index]))
    
temp = pd.Series(temp)
temp = temp.ewm(alpha=0.15).mean()
temp = temp.values

k=0
for index,value in enumerate(times):
  if(value > '2008-08-08' and value <'2008-09-01'):
    df['Global_active_power'].iloc[index] = temp[k]
    k += 1

print(f'df_shape arxika: {df.shape}\n')



train_len = int(len(df) * 0.8)
test_df = df[train_len:]
df = df[:train_len]
print(f'df_shape{df.shape}')
print(f'test_shape{test_df.shape}')
df.dropna(inplace=True)
test_df.dropna(inplace=True)


test_df = test_df.values
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(np.float64(df))
test_df = scaler.transform(np.float64(test_df))



encoder_input_data, decoder_input_data, decoder_target_data = preprocess_df(df, input_sequence_length, target_sequence_length, shuffle=True)
print(encoder_input_data.shape)



model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)
model.summary()



history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,

                   )


test_x, decoder_input_data, test_y = preprocess_df(test_df, input_sequence_length, target_sequence_length, shuffle=False)

score = model.evaluate([test_x, decoder_input_data], test_y, batch_size=batch_size)
print('Test loss evaluation:', score)


plot_history(history) 
    


encoder_predict_model = keras.models.Model(encoder_inputs, encoder_states)


decoder_states_inputs = []


for hidden_neurons in layers[::-1]:

    decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]
decoder_outputs = decoder_dense(decoder_outputs)



decoder_predict_model = keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
  

test_y_predicted = predict(test_x, encoder_predict_model, decoder_predict_model, 24)



encoder_predict_model.save('encoder.h5')
decoder_predict_model.save('decoder.h5')

encoder_predict_model.save('encoder.h5')
decoder_predict_model.save('decoder.h5')

