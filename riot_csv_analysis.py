#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "clean_data.csv"

#%% work with the new csv

df = pd.read_csv(csv_path)

df.dropna()

df = df.loc[df["challenges.gameLength"] > 1460, :]

#%% gold efficiency per role

df["team_total_damage"] = (
    df
    .groupby(["matchId", "teamId"])["totalDamageDealtToChampions"]
    .transform("sum")
)

df["damage_share"] = df["totalDamageDealtToChampions"] / df["team_total_damage"].replace(0, 1)
df["gold_share"] = df["goldEarned"] / df["team_gold"].replace(0, 1)

df["gold_efficiency"] = df["damage_share"] / df["gold_share"].replace(0, 1)

role_summary = df.groupby("teamPosition")[
    ["gold_efficiency", "damage_share", "gold_share"]
].mean().reset_index()

role_summary = role_summary.sort_values(by="gold_efficiency", ascending=False)

x = np.arange(len(role_summary["teamPosition"]))
width = 0.35

plt.bar(x - width/2, role_summary["damage_share"] * 100, width, label='Damage Share (%)', color='#e74c3c')
plt.bar(x + width/2, role_summary["gold_share"] * 100, width, label='Gold Share (%)', color='#f1c40f')

plt.ylabel('Percentage (%)')
plt.title('Damage Share vs. Gold Share by Role\n(Sorted by Gold Efficiency)')
plt.xticks(x, role_summary["teamPosition"])
plt.legend()
plt.tight_layout()
plt.show()

role_summary["damage_share"] = (role_summary["damage_share"] * 100).round(1).astype(str) + "%"
role_summary["gold_share"] = (role_summary["gold_share"] * 100).round(1).astype(str) + "%"
role_summary["gold_efficiency"] = role_summary["gold_efficiency"].round(2)

print("\n--- Average Gold Efficiency by Role ---")
print(role_summary.to_string(index=False))

#%% deaths vs kills vs assists

stats = ["kills", "deaths", "assists"]
correlations = [df[stat].corr(df["team_win"].astype(int)) for stat in stats]

colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations]

plt.bar(stats, correlations, color=colors, edgecolor='black', alpha=0.8)
plt.ylabel('Correlation Coefficient')
plt.title('Correlation of Kills, Deaths, and Assists with Winning')
plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
plt.ylim(-1.0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for i, val in enumerate(correlations):
    plt.text(i, val + (0.05 if val >= 0 else -0.1), f"{val:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
    
#%% objective importance

objectives = {
    "Dragons": "team_dragon_kills",
    "Barons": "team_baron_kills",
    "Towers": "team_tower_kills",
    "Inhibitors": "team_inhibitor_kills",
    "Grubs": "team_grubs_kills",
    "Heralds": "team_herald_kills"
}

max_values = {
    "team_dragon_kills": 6,
    "team_baron_kills": 3,
    "team_tower_kills": 11,
    "team_inhibitor_kills": 4,
    "team_grubs_kills": 6,
    "team_herald_kills": 2
}

plt.figure(figsize=(12,8))

for name, col in objectives.items():

    xs = []
    ys = []

    max_value = max_values[col]

    for value in sorted(df[col].unique()):

        if value > max_value:
            continue

        subset = df[df[col] >= value]

        xs.append(value)
        ys.append(subset["team_win"].mean())

    plt.plot(xs, ys, marker="o", label=name)

plt.xlabel("Objective Count")
plt.ylabel("Win Rate")
plt.title("Objective Importance by Win Rate")
plt.legend()
plt.grid()
plt.show()

#%% gold difference that guarantees a win

for threshold in range(1000, 20000, 1000):

    subset = df[
        df["team_gold_diff"] >= threshold
    ]

    winrate = subset["team_win"].mean()

    print(threshold, winrate)
    
    
print("gold difference of 9000 and up has a 100% win rate")
