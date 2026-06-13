#import libraries

import pandas as pd
import json

#%%Cut down json to 10000 matches

input_file = "match_data.jsonl"
output_file = "match_data_10k.jsonl"

max_matches = 10000

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    for i, line in enumerate(infile):

        if i >= max_matches:
            break

        outfile.write(line)

print("Finished.")

#%% load the json into a list

matches = []

with open(output_file, "r", encoding="utf-8") as f:
    for line in f:
        match = json.loads(line)
        matches.append(match)
print(len(matches))

# %% convert the list to a csv format

all_rows = []

for match in matches:

    match_id = match["metadata"]["matchId"]
    teams = {team["teamId"]: team for team in match["info"]["teams"]}

    for participant in match["info"]["participants"]:
        row = participant.copy()
        team = teams.get(participant["teamId"])
        row["matchId"] = match_id
        row["challenges.gameLength"] = match["info"]["gameDuration"]
        if team:
            row["team_win"] = team.get("win")
            row["team_baron_kills"] = (
                team.get("objectives", {}).get("baron", {}).get("kills", 0)
            )
            row["team_dragon_kills"] = (
                team.get("objectives", {}).get("dragon", {}).get("kills", 0)
            )
            row["team_tower_kills"] = (
                team.get("objectives", {}).get("tower", {}).get("kills", 0)
            )
            row["team_inhibitor_kills"] = (
                team.get("objectives", {}).get("inhibitor", {}).get("kills", 0)
            )
            row["team_grubs_kills"] = (
                team.get("objectives", {}).get("horde", {}).get("kills", 0)
            )
            row["team_herald_kills"] = (
                team.get("objectives", {}).get("riftHerald", {}).get("kills", 0)
            )
        else:
            row["team_win"] = None    

        all_rows.append(row)

df = pd.json_normalize(all_rows)

df.to_csv("participants.csv", index=False)