# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:15:07 2023

@author: Jahirul
"""

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
import os
import Attention_layer as Attention_layer
cwd= os.getcwd()

df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
save_dir=cwd+"\\head_pred_folder"
os.makedirs(save_dir, exist_ok=True)
print(df.isnull().any())
print(df.head())
individual_list=df['individual'].unique()
#individual_list=individual_list[:-1]
len(individual_list)
train_dir_len=int(len(individual_list)*0.75)
train_dir_list=list(individual_list[:train_dir_len])
val_dir_len=int((len(individual_list)-train_dir_len)*0.67)
val_dir_list=list(individual_list[train_dir_len:train_dir_len+val_dir_len])
test_dir_len=len(individual_list)-train_dir_len-val_dir_len
test_dir_list=list(individual_list[train_dir_len+val_dir_len:])

#Function to calculate Symmetric Mean Absolute Percentage Error
def smape(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)

training_head_df = pd.DataFrame()
val_head_df = pd.DataFrame()
test_head_df = pd.DataFrame()
count1=0
count2=0
count3=0
for i in tqdm(range(len(df))):
    csv_name = df['head'][i]
    fms = df['fms'][i]
    individual=df['individual'][i]
    simulation=df['simulation'][i]
    if(train_dir_list.count(individual)!=0):
        head_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\head\\"+csv_name
        df_head_csv = pd.read_csv(head_csv)
        df_head_csv=df_head_csv.head(28)
        try:
            df_head_csv=df_head_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_head_csv
        frames = [training_head_df, dff]
        training_head_df=pd.concat(frames)
        count1+=1
    elif(test_dir_list.count(individual)!=0):
        head_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\head\\"+csv_name
        df_head_csv = pd.read_csv(head_csv)
        df_head_csv=df_head_csv.head(28)
        try:
            df_head_csv=df_head_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_head_csv
        frames = [test_head_df, dff]
        test_head_df=pd.concat(frames)
        count2+=1
    if(val_dir_list.count(individual)!=0):
        head_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\head\\"+csv_name
        df_head_csv = pd.read_csv(head_csv)
        df_head_csv=df_head_csv.head(28)
        try:
            df_head_csv=df_head_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_head_csv=df_head_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_head_csv
        frames = [val_head_df, dff]
        val_head_df=pd.concat(frames)
        count1+=1

trainY_head=training_head_df[['fms']]
trainX_head=training_head_df.drop(columns=['fms'])

#set scaling
x_scaler_for_trainX_head=MinMaxScaler()
y_scaler_for_trainY_head=MinMaxScaler()
trainY_head=y_scaler_for_trainY_head.fit_transform(trainY_head)
trainX_head=x_scaler_for_trainX_head.fit_transform(trainX_head)

for i in tqdm(range(int((training_head_df.shape[0])/28))):
    if (i==0):
        X_train_head = np.expand_dims(trainX_head[i*28:(i+1)*28,:],0)
        Y_train_head = np.expand_dims(trainY_head[i*28],0)
    else:
        X_train_head=np.concatenate((X_train_head,np.expand_dims(trainX_head[i*28:(i+1)*28,:],0)),0)
        Y_train_head = np.concatenate((Y_train_head,np.expand_dims(trainY_head[i*28],0)),0)
    
#for sampling val datafram
val_head_df.shape
valY_head=val_head_df[['fms']]
valX_head=val_head_df.drop(columns=['fms'])

#set scaling
x_scaler_for_valX_head=MinMaxScaler()
y_scaler_for_valY_head=MinMaxScaler()
valY_head=y_scaler_for_trainY_head.transform(valY_head)
valX_head=x_scaler_for_trainX_head.transform(valX_head)

#X_val_head, Y_val_head = np.array([]),np.array([]) 
for i in tqdm(range(int((val_head_df.shape[0])/28))):
    if (i==0):
        X_val_head = np.expand_dims(valX_head[i*28:(i+1)*28,:],0)
        Y_val_head = np.expand_dims(valY_head[i*28],0)
    else:
        X_val_head=np.concatenate((X_val_head,np.expand_dims(valX_head[i*28:(i+1)*28,:],0)),0)
        Y_val_head = np.concatenate((Y_val_head,np.expand_dims(valY_head[i*28],0)),0)

#for sampling test datafram
test_head_df.shape
testY_head=test_head_df[['fms']]
testX_head=test_head_df.drop(columns=['fms'])
#set scaling
x_scaler_for_testX_head=MinMaxScaler()
y_scaler_for_testY_head=MinMaxScaler()
testY_head=y_scaler_for_trainY_head.transform(testY_head)
testX_head=x_scaler_for_trainX_head.transform(testX_head)

#X_test_head, Y_test_head = np.array([]),np.array([]) 
for i in tqdm(range(int((test_head_df.shape[0])/28))):
    if (i==0):
        X_test_head = np.expand_dims(testX_head[i*28:(i+1)*28,:],0)
        Y_test_head = np.expand_dims(testY_head[i*28],0)
    else:
        X_test_head=np.concatenate((X_test_head,np.expand_dims(testX_head[i*28:(i+1)*28,:],0)),0)
        Y_test_head = np.concatenate((Y_test_head,np.expand_dims(testY_head[i*28],0)),0)
    
print("X_train_head.shape = "+str(X_train_head.shape))
print("Y_train_head = "+str(Y_train_head.shape))
print( "X_val_head.shape = "+str(X_val_head.shape))
print( "Y_val_head = "+str(Y_val_head.shape))
print("X_test_head.shape = "+str(X_test_head.shape))
print("Y_test_head.shape = "+str(Y_test_head.shape))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train_head.shape[1], X_train_head.shape[2]), return_sequences=True, dropout=0.3))
model.add(Attention_layer.attention(return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(Y_train_head.shape[1]))
model.compile(optimizer='adam',loss='mse')
model.summary()

checkpoint_path = os.path.join(save_dir,  f'model.ckpt')
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        monitor = 'val_loss',
                                        mode = 'auto',
                                        save_best_only = True,
                                        save_weights_only=True,
                                        verbose=1)
start = time.time()
# fit the model
validation=[X_val_head,Y_val_head]
history = model.fit(X_train_head, Y_train_head,
                    epochs=50,
                    batch_size=64,
                    validation_data=validation,
                    verbose=1,
                    callbacks=[cp_callback])
Total_Time = time.time() - start
print("Total time: ", Total_Time, "seconds")

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig(os.path.join(save_dir, f'Train_val_loss.png'), dpi=300)

y_pred_head = model.predict(X_test_head)
print(y_pred_head.shape, Y_test_head.shape)
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
from scipy.stats import pearsonr
results = {}
results['dataset_name'] = "head"
results['mean_squared_error'] = round(mean_squared_error(Y_test_head[:,-1], y_pred_head[:,-1]),3)
results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test_head[:,-1], y_pred_head[:,-1])),3)
results['mean_absolute_error'] = round(mean_absolute_error(Y_test_head[:,-1], y_pred_head[:,-1]),3)
results['r2_score'] = round(r2_score(Y_test_head[:,-1], y_pred_head[:,-1]),3)
results['smape'] = round(smape(Y_test_head[:,-1], y_pred_head[:,-1]),3)
pcc, _ = pearsonr(Y_test_head[:,-1], y_pred_head[:,-1])
results['pcc']= round(pcc,3)
results['Total Time (sec)'] = round(Total_Time,3)
print(results)
result_df = pd.DataFrame(results, index=[0])
try:
    os.mkdir(save_dir)
except:
    pass
result_df.to_csv(os.path.join(save_dir, f'head_data_prediction_metrics.csv'), index = False, header=True)






















