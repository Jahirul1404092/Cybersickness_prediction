# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 16:59:45 2023

@author: Jahirul
"""

import datetime
import pandas as pd
df = pd.read_csv(r'C:\Users\Jahirul\Documents\2023-09-28_14-34-16-103119_HR.csv')
# LSL timestamp (example)
lsl_timestamp = 1695926059
lsl_time=df['LocalTimestamp']
result_localtime=pd.DataFrame()
frames=[]
for lsl in lsl_time:
    # Convert LSL timestamp to local time (UTC)
    utc_time = datetime.datetime.utcfromtimestamp(lsl)

    # Specify the target time zone (for example, 'America/New_York')
    target_timezone = 'America/New_York'
    local_timezone = datetime.timezone(datetime.timedelta(hours=-5))  # Eastern Standard Time (EST) offset is -5 hours

    # Convert UTC time to local time in the specified time zone
    local_time = utc_time.astimezone(datetime.timezone(datetime.timedelta(hours=0)))  # Convert to UTC first
    local_time = local_time.astimezone(local_timezone)  # Then convert to the target time zone
    frames.append(local_time)
    # dff=local_time
    # frames = [result_localtime, dff]
    print(local_time)
    # result_localtime=pd.concat(frames)
    #print('LSL:'+str(lsl),"-------------Local Time:", local_time)


#result_localtime = pd.DataFrame(frames, index=[1])
