# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:55:25 2023

@author: Jahirul
"""
plot_save_dir=r'C:\Users\Jahirul\Documents\dataset_fetching\metrics'
import pandas as pd
hr_df = pd.read_csv(r'C:\Users\Jahirul\Documents\dataset_fetching\hr_pred_folder\hr_data_prediction_metrics.csv')
head_df = pd.read_csv(r'C:\Users\Jahirul\Documents\dataset_fetching\head_pred_folder\head_data_prediction_metrics.csv')
eye_df = pd.read_csv(r'C:\Users\Jahirul\Documents\dataset_fetching\eye_pred_folder\eye_data_prediction_metrics.csv')
eda_df = pd.read_csv(r'C:\Users\Jahirul\Documents\dataset_fetching\eda_pred_folder\eda_data_prediction_metrics.csv')

#hr_df[['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score']][hr_df[hr_df['dataset_name']=='hr'].index.item()]
hr_metrics=hr_df.loc[hr_df['dataset_name'] == 'hr', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape']].values[0]
head_metrics=head_df.loc[head_df['dataset_name'] == 'head', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape']].values[0]
eda_metrics=eda_df.loc[eda_df['dataset_name'] == 'eda', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape']].values[0]
eye_metrics=eye_df.loc[eye_df['dataset_name'] == 'eye', ['mean_squared_error', 'root_mean_squared_error', 'mean_absolute_error', 'r2_score','smape']].values[0]

import numpy as np
import matplotlib.pyplot as plt
import os
 
# set width of bar
barWidth = 0.10
fig = plt.subplots(figsize =(16, 10))
 
# Set position of bar on X axis
br1 = np.arange(len(hr_metrics))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
 
# Make the plot 
plt.bar(br1, hr_metrics, color ='r', width = barWidth, edgecolor ='grey', label ='HR')
plt.bar(br2, head_metrics, color ='g', width = barWidth, edgecolor ='grey', label ='Head')
plt.bar(br3, eda_metrics, color ='b', width = barWidth, edgecolor ='grey', label ='EDA')
plt.bar(br4, eye_metrics, color ='black', width = barWidth, edgecolor ='grey', label ='Eye')
plt.hlines(y=0, xmin=0, xmax=3.6, colors='r')
# Adding Xticks
plt.xlabel('Type of Error', fontweight ='bold', fontsize = 15)
plt.ylabel('Metrics', fontweight ='bold', fontsize = 15)
xticks_place=[r + barWidth for r in range(len(hr_metrics))]
plt.xticks(xticks_place, ['MSE', 'RMSE', 'MAE', 'R2_score'])
plt.legend()
plt.show()
plt.savefig(os.path.join(plot_save_dir, f'Train_val_loss.png'), dpi=300)
