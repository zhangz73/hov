import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.remove('data/df_toll.csv')

for i in range(4):
    start = 3 * i + 1
    end = start + 2
    data_filename = "./NB_" + str(start).zfill(2) + "2021-" + str(end).zfill(2) + "2021.csv"
    
    df_toll = pd.read_csv(data_filename)
    
    df_toll["Zone_Toll"] = pd.to_numeric(df_toll["Zone_Toll"], errors='coerce')
    df_toll["dtMsgStartTime2"] = pd.to_datetime(df_toll["dtMsgStartTime2"])
    df_toll["Segment"] = df_toll["siZoneID"]
#    df_toll = df_toll.loc[df_toll['siZoneID'] == '3460 - Hesperian/238 NB']
    df_toll["Date"] = df_toll["dtMsgStartTime2"].dt.date
    df_toll["Hour"] = df_toll["dtMsgStartTime2"].dt.hour
    df_toll["Minute"] = df_toll["dtMsgStartTime2"].dt.minute
    df_toll["Time"] = df_toll["dtMsgStartTime2"].dt.time
    
    df_toll = df_toll[["Date", "Hour", "Segment", "Zone_Toll"]]
    df_toll = df_toll.groupby(["Date", "Hour", "Segment"]).mean().reset_index()
    df_toll = df_toll.rename(columns = {"Zone_Toll" : "Avg_total_toll"})
    
    with open('data/df_toll.csv','a') as output_file:
        should_header = i == 0
        df_toll.to_csv(output_file, header=should_header, index=False)
        print("wrote lines to output file:", len(df_toll), "from file", data_filename)
