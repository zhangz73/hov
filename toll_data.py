import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NB_042021-062021.csv")
df = df[pd.to_numeric(df["Segment_Toll"], errors='coerce').notnull()]
df = df[pd.to_numeric(df["Zone_Toll"], errors='coerce').notnull()]
df = df.dropna(subset = ["Zone_Toll"])[["dtMsgStartTime2", "siZoneID", "iPlazaID", "Zone_Toll", "Segment_Toll"]]
df = df.fillna(0)
df = df[df["siZoneID"] == "3420 - Auto Mall NB"]
df["Time"] = pd.to_datetime(df["dtMsgStartTime2"]).dt.strftime("%H:%M")
df["Toll"] = df["Zone_Toll"].astype(float) #+ df["Segment_Toll"].astype(float)
df = df[["Time", "Toll"]].groupby(["Time"]).mean().reset_index().sort_values("Time")


plt.plot(pd.to_datetime(df["Time"], format="%H:%M"), df["Toll"])
plt.gcf().autofmt_xdate()
plt.show()
