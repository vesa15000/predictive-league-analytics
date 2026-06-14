#import libraries

import pandas as pd

csv_path = "participants.csv"

#%% read the new csv we just made and drop excess columns

data_dirty = pd.read_csv(csv_path, low_memory=False)

data_clean = data_dirty.loc[:, ["assists", 
                                "challenges.gameLength",
                                "challenges.damagePerMinute", 
                                "challenges.firstTurretKilled", 
                                "challenges.flawlessAces", 
                                "challenges.goldPerMinute",
                                "challenges.hadOpenNexus",
                                "challenges.kda",
                                "challenges.teamElderDragonKills",
                                "challenges.visionScorePerMinute",
                                "champExperience",
                                "champLevel",
                                "deaths",
                                "gameEndedInSurrender",
                                "goldEarned",
                                "goldSpent",
                                "participantId",
                                "pentaKills",
                                "kills",
                                "teamId",
                                "teamPosition",
                                "totalDamageDealt",
                                "totalDamageDealtToChampions",
                                "totalDamageTaken",
                                "totalMinionsKilled",
                                "matchId",
                                "team_win",
                                "team_baron_kills",
                                "team_dragon_kills",
                                "team_tower_kills",
                                "team_inhibitor_kills",
                                "team_grubs_kills",
                                "team_herald_kills"]]

#%% make new columns for analysis (agregation and differences)

data_clean["team_gold"] = (
    data_clean
    .groupby(["matchId", "teamId"])["goldEarned"]
    .transform("sum")
)

data_clean["team_xp"] = (
    data_clean
    .groupby(["matchId", "teamId"])["champExperience"]
    .transform("sum")
)

lane_df = data_clean[
    ["matchId",
     "teamPosition",
     "teamId",
     "goldEarned",
     "champExperience"]
]

lane_opponents = lane_df.merge(
    lane_df,
    on=["matchId", "teamPosition"],
    suffixes=("", "_enemy")
)

lane_opponents = lane_opponents[
    lane_opponents["teamId"] !=
    lane_opponents["teamId_enemy"]
]

lane_opponents["lane_gold_diff"] = (
    lane_opponents["goldEarned"] -
    lane_opponents["goldEarned_enemy"]
)

lane_opponents["lane_xp_diff"] = (
    lane_opponents["champExperience"] -
    lane_opponents["champExperience_enemy"]
)

data_clean = data_clean.merge(
    lane_opponents[[
        "matchId",
        "teamId",
        "teamPosition",
        "lane_gold_diff",
        "lane_xp_diff"
    ]],
    on=["matchId", "teamId", "teamPosition"],
    how="left"
)

team_stats = data_clean[["matchId", "teamId", "team_gold", "team_xp"]].drop_duplicates()

team_matchup = team_stats.merge(
    team_stats,
    on="matchId",
    suffixes=("", "_enemy")
)

team_matchup = team_matchup[
    team_matchup["teamId"] != team_matchup["teamId_enemy"]
]

team_matchup["team_gold_diff"] = (
    team_matchup["team_gold"] - team_matchup["team_gold_enemy"]
)

team_matchup["team_xp_diff"] = (
    team_matchup["team_xp"] - team_matchup["team_xp_enemy"]
)

data_clean = data_clean.merge(
    team_matchup[["matchId", "teamId", "team_gold_diff", "team_xp_diff"]],
    on=["matchId", "teamId"],
    how="left"
)

#%% make a clean csv file

data_clean.to_csv(
    "clean_data.csv",
    index=False,
)