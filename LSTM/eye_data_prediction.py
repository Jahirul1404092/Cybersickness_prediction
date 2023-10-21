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
cwd= os.getcwd()

df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
save_dir=cwd+"\\eye_pred_folder"
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

training_eye_df = pd.DataFrame()
val_eye_df = pd.DataFrame()
test_eye_df = pd.DataFrame()
count1=0
count2=0
count3=0
for i in tqdm(range(len(df))):
    csv_name = df['eye'][i]
    fms = df['fms'][i]
    individual=df['individual'][i]
    simulation=df['simulation'][i]
    if(train_dir_list.count(individual)!=0):
        eye_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
        df_eye_csv = pd.read_csv(eye_csv)
        df_eye_csv=df_eye_csv.head(28)
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eye_csv
        frames = [training_eye_df, dff]
        training_eye_df=pd.concat(frames)
        count1+=1
    elif(test_dir_list.count(individual)!=0):
        eye_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
        df_eye_csv = pd.read_csv(eye_csv)
        df_eye_csv=df_eye_csv.head(28)
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eye_csv
        frames = [test_eye_df, dff]
        test_eye_df=pd.concat(frames)
        count2+=1
    if(val_dir_list.count(individual)!=0):
        eye_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
        df_eye_csv = pd.read_csv(eye_csv)
        df_eye_csv=df_eye_csv.head(28)
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eye_csv=df_eye_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eye_csv
        frames = [val_eye_df, dff]
        val_eye_df=pd.concat(frames)
        count1+=1

trainY_eye=training_eye_df[['fms']]
trainX_eye=training_eye_df.drop(columns=['fms'])

#set scaling
x_scaler_for_trainX_eye=MinMaxScaler()
y_scaler_for_trainY_eye=MinMaxScaler()
trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

for i in tqdm(range(int((training_eye_df.shape[0])/28))):
    if (i==0):
        X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
        Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
    else:
        X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
        Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
    
#for sampling val datafram
val_eye_df.shape
valY_eye=val_eye_df[['fms']]
valX_eye=val_eye_df.drop(columns=['fms'])

#set scaling
x_scaler_for_valX_eye=MinMaxScaler()
y_scaler_for_valY_eye=MinMaxScaler()
valY_eye=y_scaler_for_trainY_eye.transform(valY_eye)
valX_eye=x_scaler_for_trainX_eye.transform(valX_eye)

#X_val_eye, Y_val_eye = np.array([]),np.array([]) 
for i in tqdm(range(int((val_eye_df.shape[0])/28))):
    if (i==0):
        X_val_eye = np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)
        Y_val_eye = np.expand_dims(valY_eye[i*28],0)
    else:
        X_val_eye=np.concatenate((X_val_eye,np.expand_dims(valX_eye[i*28:(i+1)*28,:],0)),0)
        Y_val_eye = np.concatenate((Y_val_eye,np.expand_dims(valY_eye[i*28],0)),0)

#for sampling test datafram
test_eye_df.shape
testY_eye=test_eye_df[['fms']]
testX_eye=test_eye_df.drop(columns=['fms'])
#set scaling
x_scaler_for_testX_eye=MinMaxScaler()
y_scaler_for_testY_eye=MinMaxScaler()
testY_eye=y_scaler_for_trainY_eye.transform(testY_eye)
testX_eye=x_scaler_for_trainX_eye.transform(testX_eye)

#X_test_eye, Y_test_eye = np.array([]),np.array([]) 
for i in tqdm(range(int((test_eye_df.shape[0])/28))):
    if (i==0):
        X_test_eye = np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)
        Y_test_eye = np.expand_dims(testY_eye[i*28],0)
    else:
        X_test_eye=np.concatenate((X_test_eye,np.expand_dims(testX_eye[i*28:(i+1)*28,:],0)),0)
        Y_test_eye = np.concatenate((Y_test_eye,np.expand_dims(testY_eye[i*28],0)),0)
    
print("X_train_eye.shape = "+str(X_train_eye.shape))
print("Y_train_eye = "+str(Y_train_eye.shape))
print( "X_val_eye.shape = "+str(X_val_eye.shape))
print( "Y_val_eye = "+str(Y_val_eye.shape))
print("X_test_eye.shape = "+str(X_test_eye.shape))
print("Y_test_eye.shape = "+str(Y_test_eye.shape))

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train_eye.shape[1], X_train_eye.shape[2]), return_sequences=True, dropout=0.3))
model.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
model.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(Y_train_eye.shape[1]))
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
validation=[X_val_eye,Y_val_eye]
history = model.fit(X_train_eye, Y_train_eye,
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

y_pred_eye = model.predict(X_test_eye)
print(y_pred_eye.shape, Y_test_eye.shape)
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
from scipy.stats import pearsonr
results = {}
results['dataset_name'] = "eye"
results['mean_squared_error'] = round(mean_squared_error(Y_test_eye[:,-1], y_pred_eye[:,-1]),3)
results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test_eye[:,-1], y_pred_eye[:,-1])),3)
results['mean_absolute_error'] = round(mean_absolute_error(Y_test_eye[:,-1], y_pred_eye[:,-1]),3)
results['r2_score'] = round(r2_score(Y_test_eye[:,-1], y_pred_eye[:,-1]),3)
results['smape'] = round(smape(Y_test_eye[:,-1], y_pred_eye[:,-1]),3)
pcc, _ = pearsonr(Y_test_eye[:,-1], y_pred_eye[:,-1])
results['pcc']= round(pcc,3)
results['Total Time (sec)'] = round(Total_Time,3)
print(results)
result_df = pd.DataFrame(results, index=[0])
try:
    os.mkdir(save_dir)
except:
    pass
result_df.to_csv(os.path.join(save_dir, f'eye_data_prediction_metrics.csv'), index = False, header=True)






















