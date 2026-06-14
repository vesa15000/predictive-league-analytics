import pandas as pd

file_path = "D:/download/archive/matches.csv"
matches_path = "D:/download/archive/matches_subset_15k.csv"
#%%


df = pd.read_csv(file_path, low_memory=False)

subset = df.iloc[15000:30001]

subset.to_csv(matches_path, index=False)

#%% 

matches = pd.read_csv("matches_subset_15k.csv")



#%%

intervals_path = "D:/download/archive/intervals.csv"
processed_summoner_data_path = "D:/download/archive/processed_summoner_data.csv"

intervals = pd.read_csv(intervals_path, low_memory=False)
#%%

matches_id = matches["match_id"]
len(matches_id)

intervals_subset = intervals[intervals["match_id"].isin(matches_id)]
intervals_subset.to_csv("intervals_subset.csv",index = False)
#%%
processed_summoner_data = pd.read_csv(processed_summoner_data_path, low_memory=False)

processed_summoner_data_subset = processed_summoner_data[processed_summoner_data["match_id"].isin(matches_id)]

processed_summoner_data_subset.to_csv("processed_summoner_data_subset.csv", index = False)