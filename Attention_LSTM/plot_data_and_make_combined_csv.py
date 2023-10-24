# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:55:25 2023

@author: Jahirul
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

cwd= os.getcwd()
save_dir=cwd+"\\metrics"
os.makedirs(save_dir, exist_ok=True)


eye_df = pd.read_csv(cwd+'\\eye_pred_folder\\eye_data_prediction_metrics.csv')
head_df = pd.read_csv(cwd+'\\head_pred_folder\\head_data_prediction_metrics.csv')
eda_df = pd.read_csv(cwd+'\\eda_pred_folder\\eda_data_prediction_metrics.csv')
hr_df = pd.read_csv(cwd+'\\hr_pred_folder\\hr_data_prediction_metrics.csv')
eye_head_df = pd.read_csv(cwd+'\\eye_head_pred_folder\\eye_head_data_prediction_metrics.csv')
eye_hr_df = pd.read_csv(cwd+'\\eye_hr_pred_folder\\eye_hr_data_prediction_metrics.csv')
eye_eda_df = pd.read_csv(cwd+'\\eye_eda_pred_folder\\eye_eda_data_prediction_metrics.csv')
eda_hr_df = pd.read_csv(cwd+'\\eda_hr_pred_folder\\eda_hr_data_prediction_metrics.csv')
eye_head_eda_hr_df = pd.read_csv(cwd+'\\eye_head_eda_hr_pred_folder\\eye_head_eda_hr_data_prediction_metrics.csv')

combined_metrics_df=pd.DataFrame()
combined_metrics_df=pd.concat([eye_df, head_df, eda_df, hr_df, eye_head_df, eye_hr_df, eye_eda_df, eda_hr_df, eye_head_eda_hr_df])
combined_metrics_df.to_csv(os.path.join(save_dir, f'Combined_metrics.csv'), index = False, header=True)

eye_metrics=eye_df.loc[eye_df['dataset_name'] == 'eye', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
head_metrics=head_df.loc[head_df['dataset_name'] == 'head', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eda_metrics=eda_df.loc[eda_df['dataset_name'] == 'eda', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
hr_metrics=hr_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eye_head_metrics=eye_head_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eye_hr_metrics=eye_hr_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eye_eda_metrics=eye_eda_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eda_hr_metrics=eda_hr_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]
eye_head_eda_hr_metrics=eye_head_eda_hr_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape','pcc']].values[0]

# set width of bar
barWidth = 0.10
fig = plt.subplots(figsize =(16, 10))
 
# Set position of bar on X axis
br1 = np.arange(len(eye_metrics))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7]
br9 = [x + barWidth for x in br8]
 
# Make the plot 
plt.bar(br1, eye_metrics, color ='black', width = barWidth, edgecolor ='grey', label ='Eye')
plt.bar(br2, head_metrics, color ='g', width = barWidth, edgecolor ='grey', label ='Head')
plt.bar(br3, eda_metrics, color ='b', width = barWidth, edgecolor ='grey', label ='EDA')
plt.bar(br4, hr_metrics, color ='r', width = barWidth, edgecolor ='grey', label ='HR')
plt.bar(br5, eye_head_metrics, color ='g', width = barWidth, edgecolor ='grey', label ='Eye & Head')
plt.bar(br6, eye_hr_metrics, color ='b', width = barWidth, edgecolor ='grey', label ='Eye & HR')
plt.bar(br7, eye_eda_metrics, color ='r', width = barWidth, edgecolor ='grey', label ='Eye & EDA')
plt.bar(br8, eda_hr_metrics, color ='g', width = barWidth, edgecolor ='grey', label ='EDA & HR')
plt.bar(br9, eye_head_eda_hr_metrics, color ='b', width = barWidth, edgecolor ='grey', label ='Eye & Head & EDA & HR')

plt.hlines(y=0, xmin=0, xmax=9.6, colors='r')
# Adding Xticks
plt.xlabel('Type of Error', fontweight ='bold', fontsize = 15)
plt.ylabel('Metrics', fontweight ='bold', fontsize = 15)
xticks_place=[r + barWidth for r in range(len(eye_metrics))]
plt.xticks(xticks_place, ['MSE', 'RMSE', 'MAE', 'R2_score','SMAPE', 'PCC'])
plt.legend()
plt.show()
plt.savefig(os.path.join(save_dir, f'Error_Loss.png'), dpi=1000)
