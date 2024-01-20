import pandas as pd

df_flow = pd.read_csv('data/df_PeMs.csv')
df_toll = pd.read_csv('data/df_toll.csv')

df_meta = pd.merge(df_flow, df_toll, how = 'outer', on = ["Date", "Hour", "Segment"])
df_meta = df_meta.sort_values(by=["Date", "Hour", "Segment"], ascending=[True, True, True])
with open('data/df_meta.csv','w') as output_file:
        df_meta.to_csv(output_file, header=True, index=False)
