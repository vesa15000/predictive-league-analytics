# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

dataForAnalysis = pd.read_csv("clean_data.csv", low_memory=False)

dataForAnalysis.drop(columns='challenges.flawlessAces')

dataForAnalysis.dropna(inplace = True)




sum_cols = [
    'kills', 'deaths', 'assists', 'goldEarned', 'goldSpent', 'pentaKills', 
    'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageTaken', 'totalMinionsKilled'
]

#statistike za koje se racuna prosek
mean_cols = [
    'champLevel', 'champExperience', 'challenges.damagePerMinute', 
    'challenges.goldPerMinute', 'challenges.visionScorePerMinute', 
    'challenges.kda'
]

# statistike koje su vec na nivou tima
team_cols = [
    'team_baron_kills', 'team_dragon_kills', 'team_tower_kills', 
    'team_inhibitor_kills', 'team_grubs_kills', 'team_herald_kills', 
    'challenges.gameLength', 'challenges.flawlessAces', 
    'challenges.firstTurretKilled', 'challenges.hadOpenNexus', 
    'challenges.teamElderDragonKills', 'team_gold', 'team_xp'
]

already_diff_cols = ['lane_gold_diff', 'lane_xp_diff', 'team_gold_diff', 'team_xp_diff']

agg_dict = {}
for col in sum_cols: agg_dict[col] = 'sum'
for col in mean_cols: agg_dict[col] = 'mean'
for col in team_cols + already_diff_cols: agg_dict[col] = 'first'
agg_dict['team_win'] = 'first'

team_level = dataForAnalysis.groupby(['matchId', 'teamId']).agg(agg_dict).reset_index()

t100 = team_level[team_level['teamId'] == 100].drop(columns=['teamId'])
t200 = team_level[team_level['teamId'] == 200].drop(columns=['teamId'])

match_level = pd.merge(t100, t200, on='matchId', suffixes=('_t100', '_t200'))

df_rf = pd.DataFrame()
df_rf['matchId'] = match_level['matchId']

# racunanje razlika
all_base_cols = sum_cols + mean_cols + team_cols
for col in all_base_cols:
    df_rf[f'{col}_diff'] = match_level[f'{col}_t100'] - match_level[f'{col}_t200']

for col in already_diff_cols:
    df_rf[col] = match_level[f'{col}_t100']

df_rf['target_win'] = match_level['team_win_t100'].apply(lambda x: 1 if x in [True, 1, 'Win', 'win'] else 0)


#%%  random forest


X = df_rf.drop(columns=['matchId', 'target_win'])
y = df_rf['target_win']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Parametar': X.columns, 'Vaznost': importances})
feature_imp_df = feature_imp_df.sort_values(by='Vaznost', ascending=False)

print("\n TOP 15 PARAMETARA")
print(feature_imp_df.head(15))

top_5_features = feature_imp_df.head(5)["Parametar"].tolist()


print(f"Izabrani parametri za korelaciju: {top_5_features}")

corr_data = df_rf[top_5_features + ["target_win"]]

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


plt.title(
    "Matrica korelacije za Top 5 najvažnijih parametara i ishod meča",
    fontsize=14,
    pad=15,
)
plt.tight_layout()

plt.show()

#%% nakon gledanja matrice korelacije

X = df_rf.drop(columns=['matchId', 'target_win', 'goldEarned_diff', 'champExperience_diff', 'champLevel_diff','team_xp_diff', 'kills_diff', 'team_tower_kills_diff', 'team_inhibitor_kills_diff','challenges.goldPerMinute_diff', 'challenges.kda_diff','totalDamageDealt_diff' ,'totalDamageDealtToChampions_diff'])
y = df_rf['target_win']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_imp_df = pd.DataFrame({'Parametar': X.columns, 'Vaznost': importances})
feature_imp_df = feature_imp_df.sort_values(by='Vaznost', ascending=False)

print("\n TOP 15 PARAMETARA")
print(feature_imp_df.head(15))

top_5_features = feature_imp_df.head(5)["Parametar"].tolist()

print(f"Izabrani parametri za korelaciju: {top_5_features}")

corr_data = df_rf[top_5_features + ["target_win"]]

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

plt.title(
    "Matrica korelacije za Top 5 najvažnijih parametara i ishod meča",
    fontsize=14,
    pad=15,
)
plt.tight_layout()

plt.show()

#%%


















