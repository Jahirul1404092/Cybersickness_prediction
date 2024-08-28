# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 04:56:25 2024

@author: Jahirul
"""

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, BatchNormalization, Reshape, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, 'forecast_data', 'meta_data.csv'))
save_dir = os.path.join(cwd, "eye_pred_folder")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import matplotlib.dates as mdates

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy.signal import resample
import seaborn as sns

from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras import callbacks
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from scipy.signal import butter, filtfilt, resample
from tensorflow.keras.utils import to_categorical

base_path = os.getcwd()+'\\forecast_data'
base_path +="\\"
meta_file = base_path+ 'meta_data.csv'
meta_data = pd.read_csv(meta_file)
meta_data

def get_filtered_meta_data(individual_num, simulation_name):
  filtered_data = meta_data
  if simulation_name:
      filtered_data = filtered_data[meta_data["simulation"].isin(simulation_name)]
  if individual_num:
      filtered_data = filtered_data[meta_data["individual"].isin(individual_num)]

  return filtered_data


def sort_by_time(data):
  data['Time'] = pd.to_datetime(data['Time'], format='%H-%M-%S').dt.time
  data = data.sort_values(by='Time')
  return data

def get_data(base_path=base_path, simulation_name=None, individual_num=None,
            data_type='', sort=True):
    pruned_meta_data = get_filtered_meta_data(individual_num, simulation_name)
    file_paths = pruned_meta_data.apply(
        lambda x: '%s/%s/%s%s' % (x['individual'], x['simulation'], data_type,
                                x[data_type]),
        axis=1)
    data_list = []
    for file in file_paths:
        head_data = pd.read_csv(base_path + file)
        data_list.append(head_data)
    data = pd.concat(data_list, ignore_index=True)
    if sort:
        data = sort_by_time(data)
    return data


train_individual_list = [1,2,3,4,7,8,9,11,12,13,14]
train_simulation_list = ['sea','roller','beach','room','walk']

data_type = ['hr', 'eda', 'eye', 'head']
total_data = get_data(base_path=base_path, simulation_name=train_simulation_list,
                   individual_num=train_individual_list, data_type = 'eye')
total_data.dropna(axis=1,inplace=True)
total_data

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
unique_label=total_data['simulation'].unique()
le.fit(total_data['simulation'])
total_data['simulation_encoded'] = le.transform(total_data['simulation'])

train_individual_list = [1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 14]
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[abs(z_scores) < threshold]


def classify_fms(value, quantiles):
    if value <= quantiles[0.25]:
        return 0
    elif (value > quantiles[0.25]) & (value<=quantiles[0.75]) :
        return 1
    elif value> quantiles[0.75]:
         return 2

for individual in train_individual_list:
    sub_individual_data = total_data[total_data['individual'] == individual]
    quantiles = sub_individual_data['fms'].quantile([0.25,0.75])
    print('quantiles\n',quantiles)

    total_data.loc[total_data['individual'] == individual, 'fms_class_subjectwise'] = sub_individual_data['fms'].apply(
        lambda x: classify_fms(x, quantiles)
    )
total_data['fms_class_subjectwise'] = total_data['fms_class_subjectwise'].astype(float)  # Convert to float if necessary
total_data

total_data=remove_outliers_zscore(total_data,'fms_class_subjectwise',3)
total_data.groupby('fms_class_subjectwise').count()

total_data=total_data.drop(columns=['Convergence_distance','GazeOriginLclSpc_X','GazeOriginLclSpc_Y','GazeOriginLclSpc_Z','GazeDirectionLclSpc_X','GazeDirectionLclSpc_Y','GazeDirectionLclSpc_Z',
                                    'GazeOriginWrldSpc_X','GazeOriginWrldSpc_Y','GazeOriginWrldSpc_Z','GazeDirectionWrldSpc_X','GazeDirectionWrldSpc_Y','GazeDirectionWrldSpc_Z',
                                    'NrmLeftEyeOriginX','NrmLeftEyeOriginY','NrmLeftEyeOriginZ','NrmRightEyeOriginX','NrmRightEyeOriginY','NrmRightEyeOriginZ'])

n_hours = 28
n_features = 14
n_obj = n_features * n_hours
lowcut = 0.03
highcut = 0.3
fs_original = 1
fs_resampled = 2

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

def min_max_normalization(signal):
    scaler = MinMaxScaler()
    scaled_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    return scaled_signal, scaler

def sort_by_time(data):
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
    data = data.sort_values(by='Time')
    return data

def apply_bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def segment_data(features, target, sequence_length, simul_enc):
    X, y, z= [], [], []
    data_length = len(next(iter(features.values())))

    for i in range(int((data_length - sequence_length)/5)):
        i=i*5
        x_segment = [features[col][i:i+sequence_length] for col in features]
        X.append(np.array(x_segment).T)
        y.append(target[i + sequence_length])
        z.append(simul_enc[i + sequence_length])
    
    X_arr = np.array(X)
    y = np.array(y)
    z = np.array(z)
    return X_arr, y, z

def preprocess_data(data, sequence_length=28, columns=['Left_Eye_Openness', 'Right_Eye_Openness',
       'LeftPupilDiameter', 'RightPupilDiameter', 'LeftPupilPosInSensorX',
       'LeftPupilPosInSensorY', 'RightPupilPosInSensorX',
       'RightPupilPosInSensorY', 'NrmSRLeftEyeGazeDirX',
       'NrmSRLeftEyeGazeDirY', 'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
       'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ']):
    
    individuals = np.unique(data['individual'])
    simulations = np.unique(data['simulation'])

    X_train = []
    train_label = []
    simul_enc= []

    for participant in individuals:
        for simulation in simulations:
            df = sort_by_time(data.loc[(data['individual'] == participant) & (data['simulation'] == simulation)])
            df.index = np.arange(len(df))
            n_samples_resampled = int(len(df) * (fs_resampled / fs_original))
            features_resampled = {col: resample(df[col], n_samples_resampled) for col in columns}

            for col in columns:
                if len(features_resampled[col]) > sequence_length:  # Only apply bandpass filter if length > 60
                    features_resampled[col] = apply_bandpass_filter(features_resampled[col], lowcut, highcut, fs_resampled)
                    features_resampled[col], scaler = min_max_normalization(features_resampled[col])

            fms_list = df['fms_class_subjectwise']
            fms_doubled_list = np.array([x for x in fms_list for _ in range(int(fs_resampled/fs_original))])
            
            simul_enc_double = np.array([x for x in df['simulation_encoded'] for _ in range(int(fs_resampled/fs_original))])

            if df.shape[0] > sequence_length:
                x_arr, y, z = segment_data(features_resampled, fms_doubled_list, sequence_length, simul_enc=simul_enc_double )
                X_train.append(x_arr)
                train_label.append(y)
                simul_enc.append(z)

    X_train_whole = np.concatenate(X_train, axis=0)
    train_label_whole = np.concatenate(train_label, axis=0)
    simul_enc_whole = np.concatenate(simul_enc, axis=0)

    return X_train_whole, train_label_whole, simul_enc_whole, scaler


data = preprocess_data(total_data)
X_train_eye, Y_train_eye, simulation_encoded, x_scaler_for_trainX_eye = data[0], data[1], data[2], data[3]

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=True)
X_train_data,X_val_data, X_test_data,Y_train_data, Y_val_data, Y_test_data=[],[],[],[],[],[]


for i, (train_indices, test_indices) in enumerate(skf.split(X_train_eye, simulation_encoded)):

    x_train= X_train_eye[train_indices]
    y_train= Y_train_eye[train_indices]
    
    temp_X= X_train_eye[test_indices]
    temp_y=Y_train_eye[test_indices]
    
    x_val, x_test, y_val, y_test = train_test_split(temp_X, temp_y, test_size=0.5, random_state=42)

    
    X_train_data.append(x_train)
    Y_train_data.append(y_train)
    X_test_data.append(x_test)
    Y_test_data.append(y_test)
    X_val_data.append(x_val)
    Y_val_data.append(y_val)

X_train_list,Y_train_list,X_val_list,Y_val_list,X_test_list,Y_test_list, x_scale=X_train_data,Y_train_data,X_val_data,Y_val_data,X_test_data,Y_test_data, x_scaler_for_trainX_eye
accuracy, precision, recall, f1 = [],[],[],[]
for X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye in zip(X_train_list,Y_train_list,X_val_list,Y_val_list,X_test_list,Y_test_list):
    print(X_train_eye.shape,Y_train_eye.shape,X_val_eye.shape,Y_val_eye.shape,X_test_eye.shape,Y_test_eye.shape)
    
    Y_train_eye = Y_train_eye.astype(int)
    scaler = StandardScaler()
    X_train_eye = scaler.fit_transform(X_train_eye.reshape(-1, 28 * 14)).reshape(-1, 28, 14)
    X_val_eye = scaler.transform(X_val_eye.reshape(-1, 28 * 14)).reshape(-1, 28, 14)
    X_test_eye = scaler.transform(X_test_eye.reshape(-1, 28 * 14)).reshape(-1, 28, 14)
    
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(28, 14)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Reshape((1, 256)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(3, activation='softmax'))
    learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    class_counts = np.bincount(Y_train_eye)
    total_samples = len(Y_train_eye)
    class_weights = {i: total_samples / count for i, count in enumerate(class_counts)}

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increased patience
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_dir, 'best_model_weights.h5'), monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)  # Increased patience

    history = model.fit(X_train_eye, Y_train_eye, validation_data=(X_val_eye, Y_val_eye), epochs=150, batch_size=32, class_weight=class_weights, callbacks=[early_stopping, checkpoint, reduce_lr])

    model.load_weights(os.path.join(save_dir, 'best_model_weights.h5'))

    y_pred_eye = np.argmax(model.predict(X_test_eye), axis=1)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    a, p, r, f = accuracy_score(Y_test_eye, y_pred_eye), precision_score(Y_test_eye, y_pred_eye, average='weighted'), recall_score(Y_test_eye, y_pred_eye, average='weighted'), f1_score(Y_test_eye, y_pred_eye, average='weighted')

    print(f'Accuracy: {a:.2f}')
    print(f'Precision: {p:.2f}')
    print(f'Recall: {r:.2f}')
    print(f'F1 Score: {f:.2f}')
    accuracy.append(a), precision.append(p), recall.append(r), f1.append(f)
    tf.keras.backend.clear_session()
for i, (a,p,r,f) in enumerate(zip(accuracy, precision, recall, f1)):
    print(f'fold_{i+1}: accuracy= {a:.2f}, precision= {p:.2f}, recall= {r:.2f}, f1_score= {f:.2f}')
from statistics import mean, stdev
print(f'Mean: accuracy= {mean(accuracy):.2f}, precision= {mean(precision):.2f}, recall: {mean(recall):.2f}, F1_score: {mean(f1):.2f}')
