# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 05:54:54 2024

@author: Jahirul
"""
# import numpy as np
# 



# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:45:25 2024

@author: Jahirul
"""

import pandas as pd
import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tcn import compiled_tcn, tcn_full_summary
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[abs(z_scores) < threshold]

def getregdata():
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
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))
    
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
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        elif(test_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [test_eye_df, dff]
            test_eye_df=pd.concat(frames)
            count2+=1
        if(val_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
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
    # x_scaler_for_valX_eye=MinMaxScaler()
    # y_scaler_for_valY_eye=MinMaxScaler()
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
    # x_scaler_for_testX_eye=MinMaxScaler()
    # y_scaler_for_testY_eye=MinMaxScaler()
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
    return X_train_eye, Y_train_eye, X_val_eye, Y_val_eye, X_test_eye, Y_test_eye,x_scaler_for_trainX_eye,y_scaler_for_trainY_eye


def getclassificationXY():
    
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))

    return X_train_eye, Y_train_eye, x_scaler_for_trainX_eye, le

def getregXY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()

    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
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
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    return X_train_eye, Y_train_eye,x_scaler_for_trainX_eye,y_scaler_for_trainY_eye

def getreg_cross_XY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()

    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['CSS'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
    
    trainY_eye=training_eye_df[['fms']]
    trainX_eye=training_eye_df.drop(columns=['fms'])
    
    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    y_scaler_for_trainY_eye=MinMaxScaler()
    trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)
    
    
    # for i in tqdm(range(int(training_eye_df.shape[0]-28))):
    #     if (i==0):
    #         X_train_eye = np.expand_dims(trainX_eye[i:i+28,:],0)
    #         Y_train_eye = np.expand_dims(trainY_eye[i+28],0)
    #     else:
    #         X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i:i+28,:],0)),0)
    #         Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i+28],0)),0)
            
            
    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye.shape = "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    # X_train_data,X_test_data,Y_train_data,Y_test_data=[],[],[],[]

    # for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
    #     x_train= X_train_eye[train_indices]
    #     y_train= Y_train_eye[train_indices]
        
    #     x_test= X_train_eye[test_indices]
    #     y_test=Y_train_eye[test_indices]
        
        
    #     X_train_data.append(X_train_eye[train_indices]), X_test_data.append(X_train_eye[test_indices]),
    #     Y_train_data.append(Y_train_eye[test_indices]), Y_test_data.append(Y_train_eye[test_indices]),

    # return X_data,Y_data, x_scaler_for_trainX_eye, y_scaler_for_trainY_eye


def getclassificationdata():
    
    # from sklearn.model_selection import StratifiedKFold
    # skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    # X_data,Y_data=[],[]

    # for i, (X_indices, Y_indices) in enumerate(skf.split(train_dir_list)):
    #     X_data.append(X_train_eye[X_indices]),Y_data.append(Y_train_eye[Y_indices])
    
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
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
    # test_dir_list=[8]
    # val_dir_list=[7,9]
    #Function to calculate Symmetric Mean Absolute Percentage Error
    def smape(a, f):
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))

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
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            count1+=1
        elif(test_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [test_eye_df, dff]
            test_eye_df=pd.concat(frames)
            count2+=1
        if(val_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [val_eye_df, dff]
            val_eye_df=pd.concat(frames)
            count3+=1
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    #for sampling val datafram
    val_eye_df=val_eye_df.drop(columns=['fms'])
    val_eye_df.shape
    valY_eye=val_eye_df[['CSS']]
    valX_eye=val_eye_df.drop(columns=['CSS'])

    #set scaling
    # x_scaler_for_valX_eye=MinMaxScaler()
    # y_scaler_for_valY_eye=MinMaxScaler()
    valY_eye=le.transform(valY_eye)
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
    test_eye_df=test_eye_df.drop(columns=['fms'])
    test_eye_df.shape
    testY_eye=test_eye_df[['CSS']]
    testX_eye=test_eye_df.drop(columns=['CSS'])
    #set scaling
    # x_scaler_for_testX_eye=MinMaxScaler()
    # y_scaler_for_testY_eye=MinMaxScaler()
    testY_eye=le.transform(testY_eye)
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
    return X_train_eye, Y_train_eye, X_val_eye, Y_val_eye, X_test_eye, Y_test_eye, x_scaler_for_trainX_eye, le

def getclassification_crossfolded_XY():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','individual','simulation'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            
            #######This is for balancing classes
            '''
            if (df_eye_csv['CSS'][0]=='High'):
                frames = [training_eye_df, dff]
                training_eye_df=pd.concat(frames)
            else:
                if(df_eye_csv['CSS'][0]=='Low'):
                    count1+=1
                if(count1>=82):
                    frames = [training_eye_df, dff]
                    training_eye_df=pd.concat(frames)
            '''
        

    training_eye_df=remove_outliers_zscore(training_eye_df,'fms',3)
      
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        x_test= X_train_eye[test_indices]
        y_test=Y_train_eye[test_indices]
        
        valdata_len=int(len(x_test)*0.6)
        
        x_val=x_test[:valdata_len]
        y_val=y_test[:valdata_len]
        
        x_test=x_test[valdata_len:]
        y_test=y_test[valdata_len:]
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye, le

def getclassification_crossfolded_XY_2class():
    cwd= os.getcwd()
    df = pd.read_csv(cwd+'\\forecast_data(derived_2class)\\meta_data.csv')
    save_dir=cwd+"\\eye_pred_folder"
    os.makedirs(save_dir, exist_ok=True)
    print(df.isnull().any())
    print(df.head())
    individual_list=df['individual'].unique()
    #individual_list=individual_list[:-1]
    len(individual_list)
    train_dir_len=int(len(individual_list))
    train_dir_list=list(individual_list[:train_dir_len])

    training_eye_df = pd.DataFrame()
    val_eye_df = pd.DataFrame()
    test_eye_df = pd.DataFrame()
    count1=0
    for i in tqdm(range(len(df))):
        csv_name = df['eye'][i]
        fms = df['fms'][i]
        individual=df['individual'][i]
        simulation=df['simulation'][i]
        if(train_dir_list.count(individual)!=0):
            eye_csv=cwd+"\\forecast_data(derived_2class)\\"+str(individual)+"\\"+simulation+"\\eye\\"+csv_name
            df_eye_csv = pd.read_csv(eye_csv)
            df_eye_csv=df_eye_csv.head(28)
            try:
                df_eye_csv=df_eye_csv.drop(columns=['Time','Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ','individual','simulation','Unnamed: 40'])
            except:
                pass
            dff=df_eye_csv
            frames = [training_eye_df, dff]
            training_eye_df=pd.concat(frames)
            
            #######This is for balancing classes
            '''
            if (df_eye_csv['CSS'][0]=='High'):
                frames = [training_eye_df, dff]
                training_eye_df=pd.concat(frames)
            else:
                if(df_eye_csv['CSS'][0]=='Low'):
                    count1+=1
                if(count1>=82):
                    frames = [training_eye_df, dff]
                    training_eye_df=pd.concat(frames)
            '''
        
            
    training_eye_df=training_eye_df.drop(columns=['fms'])

    trainY_eye=training_eye_df[['CSS']]
    trainX_eye=training_eye_df.drop(columns=['CSS'])

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # unique_label=trainY_eye['CSS'].unique()
    le.fit(trainY_eye['CSS'])
    trainY_eye = le.transform(trainY_eye['CSS'])

    #set scaling
    x_scaler_for_trainX_eye=MinMaxScaler()
    # y_scaler_for_trainY_eye=MinMaxScaler()
    # trainY_eye=y_scaler_for_trainY_eye.fit_transform(trainY_eye)
    trainX_eye=x_scaler_for_trainX_eye.fit_transform(trainX_eye)

    for i in tqdm(range(int((training_eye_df.shape[0])/28))):
        if (i==0):
            X_train_eye = np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)
            Y_train_eye = np.expand_dims(trainY_eye[i*28],0)
        else:
            X_train_eye=np.concatenate((X_train_eye,np.expand_dims(trainX_eye[i*28:(i+1)*28,:],0)),0)
            Y_train_eye = np.concatenate((Y_train_eye,np.expand_dims(trainY_eye[i*28],0)),0)
        
    print("X_train_eye= "+str(X_train_eye.shape))
    print("Y_train_eye = "+str(Y_train_eye.shape))
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
    X_train_data,X_val_data, X_test_data,Y_train_data,Y_val_data, Y_test_data=[],[],[],[],[],[]
    
    for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, Y_train_eye)):
        x_train= X_train_eye[train_indices]
        y_train= Y_train_eye[train_indices]
        
        x_test= X_train_eye[test_indices]
        y_test=Y_train_eye[test_indices]
        
        valdata_len=int(len(x_test)*0.6)
        
        x_val=x_test[:valdata_len]
        y_val=y_test[:valdata_len]
        
        x_test=x_test[valdata_len:]
        y_test=y_test[valdata_len:]
        
        X_train_data.append(x_train)
        Y_train_data.append(y_train)
        X_test_data.append(x_test)
        Y_test_data.append(y_test)
        X_val_data.append(x_val)
        Y_val_data.append(y_val)
        
    return X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye, le

def get_hp_data(x_scaler_for_trainX_eye):
    cwd= os.getcwd()
    hp_eye_df = pd.read_csv(cwd+'\\EyeTrackingData2024-06-11 16-59.csv')
    try:
        hp_eye_df = hp_eye_df.drop(columns=['Timestamp', 'FrameNumber', 'LeftEye_Gaze_Confidence',
        'LeftEye_PupilPosition_Confidence', 'LeftEye_Openness_Confidence',
        'LeftEye_PupilDilation_Confidence', 'RightEye_Gaze_Confidence',
        'RightEye_PupilPosition_Confidence', 'RightEye_Openness_Confidence',
        'RightEye_PupilDilation_Confidence', 'CombinedGaze_X', 'CombinedGaze_Y',
        'CombinedGaze_Z', 'CombinedGaze_Confidence'])
        hp_eye_df=hp_eye_df.rename(columns={'LeftEye_Openness_Openness':'Left_Eye_Openness',
               'RightEye_Openness_Openness':'Right_Eye_Openness', 'LeftEye_PupilDilation_PupilDilation':'LeftPupilDiameter',
               'RightEye_PupilDilation_PupilDilation':'RightPupilDiameter', 'LeftEye_PupilPosition_X':'LeftPupilPosInSensorX',
               'LeftEye_PupilPosition_Y':'LeftPupilPosInSensorY', 'RightEye_PupilPosition_X':'RightPupilPosInSensorX',
               'RightEye_PupilPosition_Y':'RightPupilPosInSensorY', 'LeftEye_Gaze_X':'NrmSRLeftEyeGazeDirX', 'LeftEye_Gaze_Y':'NrmSRLeftEyeGazeDirY',
               'LeftEye_Gaze_Z':'NrmSRLeftEyeGazeDirZ', 'RightEye_Gaze_X':'NrmSRRightEyeGazeDirX', 'RightEye_Gaze_Y':'NrmSRRightEyeGazeDirY',
               'RightEye_Gaze_Z':'NrmSRRightEyeGazeDirZ'})
    except:
        pass

    hp_eye_df=x_scaler_for_trainX_eye.transform(hp_eye_df)

    #X_test_eye, Y_test_eye = np.array([]),np.array([]) 
    for i in tqdm(range(int((hp_eye_df.shape[0])/28))):
        if (i==0):
            X_test_eye_hp = np.expand_dims(hp_eye_df[i*28:(i+1)*28,:],0)
        else:
            X_test_eye_hp=np.concatenate((X_test_eye_hp,np.expand_dims(hp_eye_df[i*28:(i+1)*28,:],0)),0)
            
    print("X_test_eye_hp = "+str(X_test_eye_hp.shape))
    return X_test_eye_hp

def save_data(X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye):
    np.save('X_train_eye.npy', X_train_eye)
    np.save('Y_train_eye.npy', Y_train_eye)
    np.save('X_val_eye.npy', X_val_eye)
    np.save('Y_val_eye.npy', Y_val_eye)
    np.save('X_test_eye.npy',X_test_eye )
    np.save('Y_test_eye.npy', Y_test_eye)
    
def performance_metrics(Y_test_eye, y_pred_eye):
    accuracy = accuracy_score(Y_test_eye, y_pred_eye)
    precision = precision_score(Y_test_eye, y_pred_eye, average='weighted')
    recall = recall_score(Y_test_eye, y_pred_eye, average='weighted')
    f1 = f1_score(Y_test_eye, y_pred_eye, average='weighted')
    # print(f"Accuracy: {accuracy:0.2f}")
    # print(f"Precision: {precision:0.2f}")
    # print(f"Recall: {recall:0.2f}")
    # print(f"F1 Score: {f1:0.2f}")
    return accuracy, precision, recall, f1

def display_cm(Y_test_eye, y_pred_eye):
    cm = confusion_matrix(Y_test_eye, y_pred_eye)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    return cm

def Error_metrics(Y_test_eye,y_pred_eye):
    import math
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
    from scipy.stats import pearsonr
    results = {}
    results['mean_squared_error'] = round(mean_squared_error(Y_test_eye, y_pred_eye),3)
    results['root_mean_squared_error'] = round(math.sqrt(mean_squared_error(Y_test_eye, y_pred_eye)),3)
    results['mean_absolute_error'] = round(mean_absolute_error(Y_test_eye, y_pred_eye),3)
    results['r2_score'] = round(r2_score(Y_test_eye, y_pred_eye),3)
    # print(results)
    return results



if __name__=='__main__':
    getregdata()
