# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:57:26 2023

@author: Jahirul
"""

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, concatenate
from keras.models import Model, load_model
from keras.layers import Embedding, LSTM, Dropout, Dense, Input, Bidirectional, Flatten, Conv2D, MaxPooling2D, concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
from keras.utils import plot_model
import os
import Attention_layer as Attention_layer
cwd= os.getcwd()
minshape1=1000
minshape2=1000
df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
save_dir=cwd+"\\eye_head_eda_hr_pred_folder"
os.makedirs(save_dir, exist_ok=True)
print(df.isnull().any())
print(df.head())
individual_list=df['individual'].unique()
individual_list=individual_list[:-1]
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
###################################Eye start
training_eye_df = pd.DataFrame()
val_eye_df = pd.DataFrame()
test_eye_df = pd.DataFrame()
count1=0
count2_1=0
count2_2=0
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
        if(df_eye_csv.shape[0]<=minshape1):
            minshape1=df_eye_csv.shape[0]
        count2_1+=1
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
#################################################Eye end
################################################head star
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
        if(df_eye_csv.shape[0]<=minshape1):
            minshape2=df_eye_csv.shape[0]
        count2_2+=1
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
########################################## head end
###################################eda start
training_eda_df = pd.DataFrame()
val_eda_df = pd.DataFrame()
test_eda_df = pd.DataFrame()
count1=0
count2_1=0
count2_2=0
count3=0
for i in tqdm(range(len(df))):
    csv_name = df['eda'][i]
    fms = df['fms'][i]
    individual=df['individual'][i]
    simulation=df['simulation'][i]
    if(train_dir_list.count(individual)!=0):
        eda_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eda\\"+csv_name
        df_eda_csv = pd.read_csv(eda_csv)
        df_eda_csv=df_eda_csv.head(28)
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eda_csv
        frames = [training_eda_df, dff]
        training_eda_df=pd.concat(frames)
        count1+=1
    elif(test_dir_list.count(individual)!=0):
        eda_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eda\\"+csv_name
        df_eda_csv = pd.read_csv(eda_csv)
        df_eda_csv=df_eda_csv.head(28)
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eda_csv
        frames = [test_eda_df, dff]
        test_eda_df=pd.concat(frames)
        if(df_eda_csv.shape[0]<=minshape1):
            minshape1=df_eda_csv.shape[0]
        count2_1+=1
    if(val_dir_list.count(individual)!=0):
        eda_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\eda\\"+csv_name
        df_eda_csv = pd.read_csv(eda_csv)
        df_eda_csv=df_eda_csv.head(28)
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_eda_csv=df_eda_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_eda_csv
        frames = [val_eda_df, dff]
        val_eda_df=pd.concat(frames)
        count1+=1

trainY_eda=training_eda_df[['fms']]
trainX_eda=training_eda_df.drop(columns=['fms'])

#set scaling
x_scaler_for_trainX_eda=MinMaxScaler()
y_scaler_for_trainY_eda=MinMaxScaler()
trainY_eda=y_scaler_for_trainY_eda.fit_transform(trainY_eda)
trainX_eda=x_scaler_for_trainX_eda.fit_transform(trainX_eda)

for i in tqdm(range(int((training_eda_df.shape[0])/28))):
    if (i==0):
        X_train_eda = np.expand_dims(trainX_eda[i*28:(i+1)*28,:],0)
        Y_train_eda = np.expand_dims(trainY_eda[i*28],0)
    else:
        X_train_eda=np.concatenate((X_train_eda,np.expand_dims(trainX_eda[i*28:(i+1)*28,:],0)),0)
        Y_train_eda = np.concatenate((Y_train_eda,np.expand_dims(trainY_eda[i*28],0)),0)
    
#for sampling val datafram
val_eda_df.shape
valY_eda=val_eda_df[['fms']]
valX_eda=val_eda_df.drop(columns=['fms'])

#set scaling
x_scaler_for_valX_eda=MinMaxScaler()
y_scaler_for_valY_eda=MinMaxScaler()
valY_eda=y_scaler_for_trainY_eda.transform(valY_eda)
valX_eda=x_scaler_for_trainX_eda.transform(valX_eda)

#X_val_eda, Y_val_eda = np.array([]),np.array([]) 
for i in tqdm(range(int((val_eda_df.shape[0])/28))):
    if (i==0):
        X_val_eda = np.expand_dims(valX_eda[i*28:(i+1)*28,:],0)
        Y_val_eda = np.expand_dims(valY_eda[i*28],0)
    else:
        X_val_eda=np.concatenate((X_val_eda,np.expand_dims(valX_eda[i*28:(i+1)*28,:],0)),0)
        Y_val_eda = np.concatenate((Y_val_eda,np.expand_dims(valY_eda[i*28],0)),0)

#for sampling test datafram
test_eda_df.shape
testY_eda=test_eda_df[['fms']]
testX_eda=test_eda_df.drop(columns=['fms'])
#set scaling
x_scaler_for_testX_eda=MinMaxScaler()
y_scaler_for_testY_eda=MinMaxScaler()
testY_eda=y_scaler_for_trainY_eda.transform(testY_eda)
testX_eda=x_scaler_for_trainX_eda.transform(testX_eda)

#X_test_eda, Y_test_eda = np.array([]),np.array([]) 
for i in tqdm(range(int((test_eda_df.shape[0])/28))):
    if (i==0):
        X_test_eda = np.expand_dims(testX_eda[i*28:(i+1)*28,:],0)
        Y_test_eda = np.expand_dims(testY_eda[i*28],0)
    else:
        X_test_eda=np.concatenate((X_test_eda,np.expand_dims(testX_eda[i*28:(i+1)*28,:],0)),0)
        Y_test_eda = np.concatenate((Y_test_eda,np.expand_dims(testY_eda[i*28],0)),0)
    
print("X_train_eda.shape = "+str(X_train_eda.shape))
print("Y_train_eda = "+str(Y_train_eda.shape))
print( "X_val_eda.shape = "+str(X_val_eda.shape))
print( "Y_val_eda = "+str(Y_val_eda.shape))
print("X_test_eda.shape = "+str(X_test_eda.shape))
print("Y_test_eda.shape = "+str(Y_test_eda.shape))
#################################################eda end
################################################Hr star
training_hr_df = pd.DataFrame()
val_hr_df = pd.DataFrame()
test_hr_df = pd.DataFrame()
count1=0
count2=0
count3=0
for i in tqdm(range(len(df))):
    csv_name = df['hr'][i]
    fms = df['fms'][i]
    individual=df['individual'][i]
    simulation=df['simulation'][i]
    if(train_dir_list.count(individual)!=0):
        hr_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\hr\\"+csv_name
        df_hr_csv = pd.read_csv(hr_csv)
        df_hr_csv=df_hr_csv.head(28)
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_hr_csv
        frames = [training_hr_df, dff]
        training_hr_df=pd.concat(frames)
        count1+=1
    elif(test_dir_list.count(individual)!=0):
        hr_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\hr\\"+csv_name
        df_hr_csv = pd.read_csv(hr_csv)
        df_hr_csv=df_hr_csv.head(28)
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_hr_csv
        frames = [test_hr_df, dff]
        test_hr_df=pd.concat(frames)
        if(df_eda_csv.shape[0]<=minshape1):
            minshape2=df_eda_csv.shape[0]
        count2_2+=1
    if(val_dir_list.count(individual)!=0):
        hr_csv=cwd+"\\forecast_data\\"+str(individual)+"\\"+simulation+"\\hr\\"+csv_name
        df_hr_csv = pd.read_csv(hr_csv)
        df_hr_csv=df_hr_csv.head(28)
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Time'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['individual'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['simulation'])
        except:
            pass
        try:
            df_hr_csv=df_hr_csv.drop(columns=['Unnamed: 40'])
        except:
            pass
        dff=df_hr_csv
        frames = [val_hr_df, dff]
        val_hr_df=pd.concat(frames)
        count1+=1

trainY_hr=training_hr_df[['fms']]
trainX_hr=training_hr_df.drop(columns=['fms'])

#set scaling
x_scaler_for_trainX_hr=MinMaxScaler()
y_scaler_for_trainY_hr=MinMaxScaler()
trainY_hr=y_scaler_for_trainY_hr.fit_transform(trainY_hr)
trainX_hr=x_scaler_for_trainX_hr.fit_transform(trainX_hr)

for i in tqdm(range(int((training_hr_df.shape[0])/28))):
    if (i==0):
        X_train_hr = np.expand_dims(trainX_hr[i*28:(i+1)*28,:],0)
        Y_train_hr = np.expand_dims(trainY_hr[i*28],0)
    else:
        X_train_hr=np.concatenate((X_train_hr,np.expand_dims(trainX_hr[i*28:(i+1)*28,:],0)),0)
        Y_train_hr = np.concatenate((Y_train_hr,np.expand_dims(trainY_hr[i*28],0)),0)
    
#for sampling val datafram
val_hr_df.shape
valY_hr=val_hr_df[['fms']]
valX_hr=val_hr_df.drop(columns=['fms'])

#set scaling
x_scaler_for_valX_hr=MinMaxScaler()
y_scaler_for_valY_hr=MinMaxScaler()
valY_hr=y_scaler_for_trainY_hr.transform(valY_hr)
valX_hr=x_scaler_for_trainX_hr.transform(valX_hr)

#X_val_hr, Y_val_hr = np.array([]),np.array([]) 
for i in tqdm(range(int((val_hr_df.shape[0])/28))):
    if (i==0):
        X_val_hr = np.expand_dims(valX_hr[i*28:(i+1)*28,:],0)
        Y_val_hr = np.expand_dims(valY_hr[i*28],0)
    else:
        X_val_hr=np.concatenate((X_val_hr,np.expand_dims(valX_hr[i*28:(i+1)*28,:],0)),0)
        Y_val_hr = np.concatenate((Y_val_hr,np.expand_dims(valY_hr[i*28],0)),0)

#for sampling test datafram
test_hr_df.shape
testY_hr=test_hr_df[['fms']]
testX_hr=test_hr_df.drop(columns=['fms'])
#set scaling
x_scaler_for_testX_hr=MinMaxScaler()
y_scaler_for_testY_hr=MinMaxScaler()
testY_hr=y_scaler_for_trainY_hr.transform(testY_hr)
testX_hr=x_scaler_for_trainX_hr.transform(testX_hr)

#X_test_hr, Y_test_hr = np.array([]),np.array([]) 
for i in tqdm(range(int((test_hr_df.shape[0])/28))):
    if (i==0):
        X_test_hr = np.expand_dims(testX_hr[i*28:(i+1)*28,:],0)
        Y_test_hr = np.expand_dims(testY_hr[i*28],0)
    else:
        X_test_hr=np.concatenate((X_test_hr,np.expand_dims(testX_hr[i*28:(i+1)*28,:],0)),0)
        Y_test_hr = np.concatenate((Y_test_hr,np.expand_dims(testY_hr[i*28],0)),0)
    
print("X_train_hr.shape = "+str(X_train_hr.shape))
print("Y_train_hr = "+str(Y_train_hr.shape))
print( "X_val_hr.shape = "+str(X_val_hr.shape))
print( "Y_val_hr = "+str(Y_val_hr.shape))
print("X_test_hr.shape = "+str(X_test_hr.shape))
print("Y_test_hr.shape = "+str(Y_test_hr.shape))
########################################## Hr end

model1 = Sequential()
model1.add(LSTM(128, activation='relu', input_shape=(X_train_eye.shape[1], X_train_eye.shape[2]), return_sequences=True, dropout=0.3))
model1.add(Attention_layer.attention(return_sequences=True))
model1.add(LSTM(64, activation='relu', return_sequences=True, dropout=0.5))
model1.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
#model1.add(Attention_layer.attention(return_sequences=True))
model1.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model1.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
#model1.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model1.add(Dense(10))
#model1.add(Dense(Y_train_eye.shape[1]))
model1.summary()

model2 = Sequential()
model2.add(LSTM(128, activation='relu', input_shape=(X_train_head.shape[1], X_train_head.shape[2]), return_sequences=True, dropout=0.3))
model2.add(Attention_layer.attention(return_sequences=True))
model2.add(LSTM(64, activation='relu', return_sequences=True, dropout=0.5))
model2.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
#model2.add(Attention_layer.attention(return_sequences=True))
model2.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model2.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
#model2.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model2.add(Dense(10))
#model2.add(Dense(Y_train_head.shape[1]))
model2.summary()

model3 = Sequential()
model3.add(LSTM(128, activation='relu', input_shape=(X_train_eda.shape[1], X_train_eda.shape[2]), return_sequences=True, dropout=0.3))
model3.add(Attention_layer.attention(return_sequences=True))
model3.add(LSTM(64, activation='relu', return_sequences=True, dropout=0.5))
model3.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
#model3.add(Attention_layer.attention(return_sequences=True))
model3.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model3.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
#model1.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model1.add(Dense(10))
#model1.add(Dense(Y_train_eye.shape[1]))
model3.summary()

model4 = Sequential()
model4.add(LSTM(128, activation='relu', input_shape=(X_train_hr.shape[1], X_train_hr.shape[2]), return_sequences=True, dropout=0.3))
model4.add(Attention_layer.attention(return_sequences=True))
model4.add(LSTM(64, activation='relu', return_sequences=True, dropout=0.5))
model4.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
#model4.add(Attention_layer.attention(return_sequences=True))
model4.add(LSTM(32, activation='relu', return_sequences=True, dropout=0.5))
model4.add(LSTM(16, activation='relu', return_sequences=True, dropout=0.5))
#model2.add(LSTM(10, activation='relu', return_sequences=False, dropout=0.5))
#model2.add(Dense(10))
#model2.add(Dense(Y_train_head.shape[1]))
model4.summary()

merged_output=concatenate([model1.output, model2.output, model3.output, model4.output])

fully_connected = Dense(16, activation='relu')(merged_output)
output_layer= LSTM(16, activation='relu', return_sequences=False, dropout=0.5)(fully_connected)
output_layer = Dense(8, activation='relu')(output_layer)
output= Dense(1, activation='relu')(output_layer)
combined_model = Model([model1.input, model2.input, model3.input, model4.input], output)
combined_model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
combined_model.summary()
plot_model(combined_model, to_file=save_dir+'\model_architecture.png',show_shapes=True)
checkpoint_path = os.path.join(save_dir,  f'model.ckpt')
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        monitor = 'val_loss',
                                        mode = 'auto',
                                        save_best_only = True,
                                        save_weights_only=True,
                                        verbose=1)
start = time.time()
history = combined_model.fit([X_train_eye, X_train_head, X_train_eda, X_train_hr], y=Y_train_head, epochs=100, batch_size=64, validation_data=([X_val_eye,X_val_head, X_val_eda,X_val_hr], Y_val_head), verbose=1, callbacks=[cp_callback])
Total_Time = time.time() - start
print("Total time: ", Total_Time, "seconds")

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig(os.path.join(save_dir, f'Train_val_loss.png'), dpi=300)

y_pred_eye_head_eda_hr = combined_model.predict([X_test_eye, X_test_head, X_test_eda, X_test_hr])
print(y_pred_eye_head_eda_hr.shape, Y_test_eye.shape)
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
from scipy.stats import pearsonr
results = {}
results['dataset_name'] = "eye_head_eda_hr"
results['mean_squared_error'] = round(mean_squared_error(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1]),3)
results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1])),3)
results['mean_absolute_error'] = round(mean_absolute_error(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1]),3)
results['r2_score'] = round(r2_score(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1]),3)
results['smape'] = round(smape(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1]),3)
pcc, _ = pearsonr(Y_test_eye[:,-1], y_pred_eye_head_eda_hr[:,-1])
results['pcc']= round(pcc,3)
results['Total Time (sec)'] = round(Total_Time,3)
print(results)

result_df = pd.DataFrame(results, index=[0])
try:
    os.mkdir(save_dir)
except:
    pass
result_df.to_csv(os.path.join(save_dir, f'eye_head_eda_hr_data_prediction_metrics.csv'), index = False, header=True)