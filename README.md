# predictive-league-analytics
Extracting and analyzing 10,000+ League of Legends matches via the Riot API to uncover objective value, map lane importance, and predict game results.
# Uputstvo za skripte
## json_process.py
Promeniti **json_path** na putanju gde ste skinuli **match_data.jsonl** ako je potrebno. Ova skripta će konvertovati jsonl u csv sa 10000 mečeva. Link do json fajla: https://www.kaggle.com/datasets/californianbill/patch-25-14-lol-league-of-legends-ranked-games/data?select=match_data.jsonl

## dirty_csv_process.py
Promeniti **csv_path** na putanju gde ste generisali **participants.csv** ako je potrebno. Ova skripta će dodatno filtrirati csv generisan u prvoj skripti i dodati agregirane podatke i razlike.

## riot_csv_process.py
Promeniti **csv_path** na putanju gde ste generisali **clean_data.csv** ako je potrebno. Ova skripta će izvršiti generalnu analizu nad odabranim temama i parametrima i ispisati grafike i rezultate za njih.

## random_forrest_finished_matches.py
Promeniti **csv_path** na putanju gde ste generisali **clean_data.csv** ako je potrebno. Ova skripta će spremiti podatke za radnom forrest algoritam, izvršiti isti i prikazati rang najbitnijih parametara.

## create_subset.py
Promeniti **file_path** na putanju gde ste skinuli **matches.csv**. Link do csv fajla: https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026?select=matches.csv

Promeniti **intervals_path** na putanju gde ste skinuli **intervals.csv**. Link do csv fajla: https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026?select=intervals.csv

Promeniti **processed_summoner_data_path** na putanju gde ste skinuli **processed_summoner_data.csv**. Link do csv fajla: https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026?select=processed_summoner_data.csv
