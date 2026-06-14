
#%%# 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd



matches = pd.read_csv("matches_subset_15k.csv")

#%%
intervals_subset = pd.read_csv("intervals_subset.csv")
processed_summoner_data_subset = pd.read_csv("processed_summoner_data_subset.csv")


intervals_subset_5min = intervals_subset.loc[intervals_subset["minute"] == 5]
intervals_subset_10min = intervals_subset.loc[intervals_subset["minute"] == 10]
intervals_subset_15min = intervals_subset.loc[intervals_subset["minute"] == 15]
intervals_subset_20min = intervals_subset.loc[intervals_subset["minute"] == 20]

print(intervals_subset_5min["match_id"].nunique())
print()
print(intervals_subset_10min["match_id"].nunique())
print()
print(intervals_subset_15min["match_id"].nunique())
print()
print(intervals_subset_20min["match_id"].nunique())


valid_matches = matches[matches["match_id"].isin(intervals_subset_20min["match_id"])]
#trajanje duze od 25min
valid_matches = valid_matches.loc[valid_matches["game_duration"] > 1499]
valid_intervals_subset = intervals_subset[intervals_subset["match_id"].isin(valid_matches["match_id"])]
valid_processed_summoner_data_subset = processed_summoner_data_subset[processed_summoner_data_subset["match_id"].isin(valid_matches["match_id"])]

#%%
intervals_subset_5min = valid_intervals_subset.loc[intervals_subset["minute"] == 5]
intervals_subset_10min = valid_intervals_subset.loc[intervals_subset["minute"] == 10]
intervals_subset_15min = valid_intervals_subset.loc[intervals_subset["minute"] == 15]
intervals_subset_20min = valid_intervals_subset.loc[intervals_subset["minute"] == 20]


print(intervals_subset_5min["match_id"].nunique())
print()
print(intervals_subset_10min["match_id"].nunique())
print()
print(intervals_subset_15min["match_id"].nunique())
print()
print(intervals_subset_20min["match_id"].nunique())



def pripremi_podatke_za_interval(df_interval, df_summoner, df_matches):
    
    df_interval = df_interval.copy()
    df_summoner = df_summoner.copy()
    df_matches = df_matches.copy()
    
    df_interval['match_id'] = df_interval['match_id'].astype(str).str.strip()
    df_summoner['match_id'] = df_summoner['match_id'].astype(str).str.strip()
    df_matches['match_id'] = df_matches['match_id'].astype(str).str.strip()
    
    df_interval['player_id'] = pd.to_numeric(df_interval['player_id'], errors='coerce')
    df_summoner['id'] = pd.to_numeric(df_summoner['id'], errors='coerce')
    

    
    timske_kolone_iz_summoner = [
        'match_id', 'id', 'team_id', 'team_kills', 'team_inhibitors', 'team_towers', 
        'team_dragons', 'team_barons', 'team_void_grubs', 'team_heralds', 'team_gold_diff'
    ]
    postojane_kolone = [col for col in timske_kolone_iz_summoner if col in df_summoner.columns]
    
    merged_df = pd.merge(
        df_interval,
        df_summoner[postojane_kolone],
        left_on=['match_id', 'player_id'],
        right_on=['match_id', 'id'],
        how='inner'
    )
    
    if 'id' in merged_df.columns:
        merged_df = merged_df.drop(columns=['id'])
    
    final_merged = pd.merge(
        merged_df,
        df_matches[['match_id', 'winning_team']],
        on='match_id',
        how='inner'
    )
    
    return final_merged

def agregiraj_na_nivo_tima(merged_df):
    sum_cols = ['total_gold', 'cs', 'jungle_cs', 'xp', 'kills', 'deaths', 'assists']
    mean_cols = ['level']
    team_cols = [
        'team_kills', 'team_inhibitors', 'team_towers', 'team_dragons',
        'team_barons', 'team_void_grubs', 'team_heralds', 'team_gold_diff'
    ]

    agg_dict = {col: 'sum' for col in sum_cols}
    agg_dict.update({col: 'mean' for col in mean_cols})
    agg_dict.update({col: 'first' for col in team_cols})
    agg_dict['winning_team'] = 'first' 

    team_level = merged_df.groupby(['match_id', 'team_id']).agg(agg_dict).reset_index()
    return team_level, sum_cols, mean_cols, team_cols

def napravi_rf_dataset(team_level, sum_cols, mean_cols, team_cols):
    t100 = team_level[team_level['team_id'] == 100].drop(columns=['team_id'])
    t200 = team_level[team_level['team_id'] == 200].drop(columns=['team_id'])

    match_level = pd.merge(t100, t200, on='match_id', suffixes=('_t100', '_t200'))

    df_rf = pd.DataFrame()
    df_rf['match_id'] = match_level['match_id']

    # Racunanje razlika
    for col in sum_cols + mean_cols + [c for c in team_cols if c != 'team_gold_diff']:
        df_rf[f'{col}_diff'] = match_level[f'{col}_t100'] - match_level[f'{col}_t200']

    df_rf['team_gold_diff'] = match_level['team_gold_diff_t100']

    df_rf['target_win'] = (match_level['winning_team_t100'] == 100).astype(int)

    return df_rf






merged_5min = pripremi_podatke_za_interval(intervals_subset_5min, processed_summoner_data_subset, matches)
team_level_5min, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_5min)
df_rf_5 = napravi_rf_dataset(team_level_5min, sum_cols, mean_cols, team_cols)

#%%
merged_10min = pripremi_podatke_za_interval(intervals_subset_10min, processed_summoner_data_subset, matches)
team_level_10min, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_10min)
df_rf_10 = napravi_rf_dataset(team_level_10min, sum_cols, mean_cols, team_cols)

#%%
merged_15min = pripremi_podatke_za_interval(intervals_subset_15min, processed_summoner_data_subset, matches)
team_level_15min, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_15min)
df_rf_15 = napravi_rf_dataset(team_level_15min, sum_cols, mean_cols, team_cols)
#%%
merged_20min = pripremi_podatke_za_interval(intervals_subset_20min, processed_summoner_data_subset, matches)
team_level_20min, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_20min)
df_rf_20 = napravi_rf_dataset(team_level_20min, sum_cols, mean_cols, team_cols)
#%%

def analiziraj_i_vizuelizuj(model, X_data, df_izvor, vreme_label):
    
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Parametar': X_data.columns, 'Vaznost': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Vaznost', ascending=False)

    print(f"\n" + "="*60)
    print(f" TOP 10 PARAMETARA ZA: {vreme_label.upper()} ".center(60, "="))
    print("="*60)
    print(feature_imp_df.head(10).to_string(index=False))

    top_5_features = feature_imp_df.head(5)["Parametar"].tolist()
    
    corr_data = df_izvor[top_5_features + ["target_win"]]
    corr_matrix = corr_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=1,
        vmax=1,
        vmin=-1,
    )
    plt.title(f"Matrica korelacije (Top 5 parametara) - {vreme_label}", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
#%%
df_rf_5.dropna(inplace=True)



# RANDOM FOREST 5MIN


X_5 = df_rf_5.drop(columns=['match_id', 'target_win', 'total_gold_diff'])
y = df_rf_5['target_win']

X_train, X_test, y_train, y_test = train_test_split(X_5, y, test_size=0.2, random_state=42)

rf_5 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_5.fit(X_train, y_train)

# Brza provera tačnosti
accuracy = rf_5.score(X_test, y_test)
print(f"Tačnost modela na test setu (5. minut): {accuracy:.2%}")

analiziraj_i_vizuelizuj(rf_5, X_5, df_rf_5, "5. minut")

#%%

df_rf_10.dropna(inplace=True)


# RANDOM FOREST 10MIN

X_10 = df_rf_10.drop(columns=['match_id', 'target_win', 'xp_diff', 'total_gold_diff'])
y = df_rf_10['target_win']

X_train, X_test, y_train, y_test = train_test_split(X_10, y, test_size=0.2, random_state=42)

rf_10 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_10.fit(X_train, y_train)

accuracy = rf_10.score(X_test, y_test)
print(f"Tačnost modela na test setu (10. minut): {accuracy:.2%}")

analiziraj_i_vizuelizuj(rf_10, X_10, df_rf_10, "10. minut")


#%%


df_rf_15.dropna(inplace=True)


# RANDOM FOREST 15MIN

X_15 = df_rf_15.drop(columns=['match_id', 'target_win', 'xp_diff', 'total_gold_diff', 'level_diff'])
y = df_rf_15['target_win']

X_train, X_test, y_train, y_test = train_test_split(X_15, y, test_size=0.2, random_state=42)

rf_15 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_15.fit(X_train, y_train)

accuracy = rf_15.score(X_test, y_test)
print(f"Tačnost modela na test setu (15. minut): {accuracy:.2%}")

analiziraj_i_vizuelizuj(rf_15, X_15, df_rf_15, "15. minut")


#%%
df_rf_20.dropna(inplace=True)


# RANDOM FOREST 20MIN

X_20 = df_rf_20.drop(columns=['match_id', 'target_win','xp_diff', 'total_gold_diff', 'level_diff', 'team_kills_diff', 'kills_diff'])
y = df_rf_20['target_win']

X_train, X_test, y_train, y_test = train_test_split(X_20, y, test_size=0.2, random_state=42)

rf_20 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_20.fit(X_train, y_train)

accuracy = rf_20.score(X_test, y_test)
print(f"Tačnost modela na test setu (15. minut): {accuracy:.2%}")

analiziraj_i_vizuelizuj(rf_20, X_20, df_rf_20, "20. minut")

#%%















