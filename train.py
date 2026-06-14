import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
#%%

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

    for col in sum_cols + mean_cols + [c for c in team_cols if c != 'team_gold_diff']:
        df_rf[f'{col}_diff'] = match_level[f'{col}_t100'] - match_level[f'{col}_t200']

    df_rf['team_gold_diff'] = match_level['team_gold_diff_t100']

    df_rf['target_win'] = (match_level['winning_team_t100'] == 100).astype(int)

    return df_rf



#%%


top_5_po_minutima = {
    "5. minut": [
        'team_gold_diff', 'xp_diff', 'cs_diff', 
        'jungle_cs_diff', 'assists_diff'
    ],
    "10. minut": [
        'team_gold_diff', 'cs_diff', 'jungle_cs_diff', 
        'assists_diff', 'level_diff'
    ],
    "15. minut": [
        'team_gold_diff', 'cs_diff', 'jungle_cs_diff', 
        'assists_diff', 'deaths_diff'
    ],
    "20. minut": [
        'team_gold_diff', 'jungle_cs_diff', 'deaths_diff', 
        'cs_diff', 'assists_diff'
    ]
}

minuti_podaci = {
    "5. minut": intervals_subset_5min,
    "10. minut": intervals_subset_10min,
    "15. minut": intervals_subset_15min,
    "20. minut": intervals_subset_20min
}

neuronske_mreze = {}
skaleri_po_minutu = {}
istorija_treninga = {}

#treniranje

for minut, trenutni_subset in minuti_podaci.items():
    print("\n" + "="*60)
    print(f" KREIRANJE NEURONSKE MREŽE ZA: {minut.upper()} ".center(60, "="))
    print("="*60)
    
    trenutni_parametri = top_5_po_minutima[minut]
    print(f"Korišćeni parametri za ovaj model: {trenutni_parametri}")
    
    merged_df = pripremi_podatke_za_interval(trenutni_subset, processed_summoner_data_subset, matches)
    team_level, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_df)
    df_rf_trenutni = napravi_rf_dataset(team_level, sum_cols, mean_cols, team_cols)
    
    df_rf_trenutni.dropna(inplace=True)
    
    X = df_rf_trenutni[trenutni_parametri]
    y = df_rf_trenutni['target_win']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    skaleri_po_minutu[minut] = scaler
    
    model = Sequential([
        Dense(16, activation='relu', input_shape=(5,)), 
        Dense(8, activation='relu'),                    
        Dense(1, activation='sigmoid')                  
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print(f"\nZapočinjem trening za {minut}...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    neuronske_mreze[minut] = model
    istorija_treninga[minut] = history
    
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\n--- REZULTAT ZA {minut.upper()} ---")
    print(f"Tačnost modela na test setu sa njegovih top 5 parametara: {accuracy:.2%}")

print("\n" + "="*60)
print(" sva 4 modela su istrenirana ".center(60))
print("="*60)

#%% ROC AUC


zbirni_izvestaj = []


for minut, trenutni_subset in minuti_podaci.items():
    trenutni_parametri = top_5_po_minutima[minut]
    
    merged_df = pripremi_podatke_za_interval(trenutni_subset, processed_summoner_data_subset, matches)
    team_level, sum_cols, mean_cols, team_cols = agregiraj_na_nivo_tima(merged_df)
    df_rf_trenutni = napravi_rf_dataset(team_level, sum_cols, mean_cols, team_cols)
    df_rf_trenutni.dropna(inplace=True)
    
    X = df_rf_trenutni[trenutni_parametri]
    y = df_rf_trenutni['target_win']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = skaleri_po_minutu[minut]
    model = neuronske_mreze[minut]
    
    X_test_scaled = scaler.transform(X_test)
    
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    y_pred_proba = model.predict(X_test_scaled).ravel()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    zbirni_izvestaj.append({
        "Vreme": minut,
        "Test Loss": round(loss, 4),
        "Test Accuracy (%)": round(accuracy * 100, 2),
        "ROC-AUC Score": round(roc_auc, 4)
    })

df_validacija = pd.DataFrame(zbirni_izvestaj)


print("\n" + "="*50)
print(" UPORDNI PREGLED KVALITETA MODELA ".center(50, "#"))
print("="*50)
print(df_validacija.to_string(index=False))
print("="*50)


plt.figure(figsize=(10, 5))

plt.plot(df_validacija["Vreme"], df_validacija["Test Accuracy (%)"], marker='o', linewidth=2, color='blue', label='Accuracy (%)')
plt.plot(df_validacija["Vreme"], df_validacija["ROC-AUC Score"] * 100, marker='s', linewidth=2, color='orange', linestyle='--', label='ROC-AUC (x100)')

plt.title("Validacija modela", fontsize=12, pad=15)
plt.xlabel("Period meča", fontsize=10)
plt.ylabel("Skala uspešnosti", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


