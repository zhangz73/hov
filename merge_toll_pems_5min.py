import pandas as pd

df_flow = pd.read_csv('data/df_PeMs_5min.csv')
df_toll = pd.read_csv('data/df_toll_5min.csv')

df_flow["Date"] = pd.to_datetime(df_flow["Date"]).dt.strftime("%Y-%m-%d")
df_toll["Date"] = pd.to_datetime(df_toll["Date"]).dt.strftime("%Y-%m-%d")

df_meta = pd.merge(df_flow, df_toll, how = 'outer', on = ["Date", "Hour", "Minute", "Segment"])
df_meta = df_meta.sort_values(by=["Date", "Hour", "Minute", "Segment"], ascending=[True, True, True, True])
with open('data/df_meta_5min.csv','w') as output_file:
        df_meta.to_csv(output_file, header=True, index=False)
