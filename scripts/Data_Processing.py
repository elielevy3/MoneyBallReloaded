# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import time
import pandas as pd
from unidecode import unidecode

csv_files_location = "/home/elie/Documents/MoneyBallReloaded/csv/"


def clean_names(df, col_name):
    df[col_name] = df[col_name].apply(str.replace, args=[" Jr.", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" Sr.", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" III", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" II", ""])
    df[col_name] = df[col_name].apply(unidecode)
    df[col_name] = df[col_name] = df[col_name].apply(str.replace, args=[".", ""])
    return df


# latest data
advance_2021 = pd.read_csv(csv_files_location + 'nba2021_advanced.csv')
per_36_minutes_2021 = pd.read_csv(csv_files_location + 'nba2021_per36min.csv')
per_game_2021 = pd.read_csv(csv_files_location + 'nba2021_per_game.csv')

test = per_game_2021["G"].sort_values()

# retrieving basic stats
df_2016 = pd.read_csv(csv_files_location + 'NBA_totals_2015-2016.csv')
df_2017 = pd.read_csv(csv_files_location + 'NBA_totals_2016-2017.csv')
df_2018 = pd.read_csv(csv_files_location + 'NBA_totals_2017-2018.csv')
df_2019 = pd.read_csv(csv_files_location + 'NBA_totals_2018-2019.csv')
df_2020 = pd.read_csv(csv_files_location + 'NBA_totals_2019-2020.csv')

df_2016 = clean_names(df_2016, "Player")
df_2017 = clean_names(df_2017, "Player")
df_2018 = clean_names(df_2018, "Player")
df_2019 = clean_names(df_2019, "Player")
df_2020 = clean_names(df_2020, "Player")

# we get the final team of each player for a given year
# it's 2 birds one rock because we can filter on retired players as well
team_and_player = df_2020[["Player", "Tm", 'Pos']]
team_and_player["final_team"] = team_and_player.groupby('Player')['Tm'].transform('last')
team_and_player = team_and_player[["Player", "final_team", "Pos"]]
team_and_player = team_and_player.drop_duplicates(subset=['Player'])

# we remove the TOT lines for the players who have been traded during the season
df_2016 = df_2016[df_2016["Tm"] != "TOT"]
df_2017 = df_2017[df_2017["Tm"] != "TOT"]
df_2018 = df_2018[df_2018["Tm"] != "TOT"]
df_2019 = df_2019[df_2019["Tm"] != "TOT"]
df_2020 = df_2020[df_2020["Tm"] != "TOT"]

# we only keep the cols we are interested in
basic_stats_2016 = df_2016.loc[:,
                   ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS']]
basic_stats_2017 = df_2017.loc[:,
                   ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS']]
basic_stats_2018 = df_2018.loc[:,
                   ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS']]
basic_stats_2019 = df_2019.loc[:,
                   ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS']]
basic_stats_2020 = df_2020.loc[:,
                   ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST',
                    'STL', 'BLK', 'TOV', 'PF', 'PTS']]

# we concat row-wise
basic_stats = basic_stats_2016.append(basic_stats_2017).append(basic_stats_2018).append(basic_stats_2019).append(
    basic_stats_2020)

# group by player name
summed_basic_stats = basic_stats.groupby(['Player']).sum()

# to avoid having outliers when projecting on 36 mn base, we remove players who did not play enough
summed_basic_stats = summed_basic_stats.loc[(summed_basic_stats['G'] > 30) | (summed_basic_stats['MP'] > 500)]


# round up to a certain nb of decimals
def custom_round_up(x, y):
    return round(x, y)


avg_stats = summed_basic_stats.loc[:,
            (summed_basic_stats.columns != "Player") & (summed_basic_stats.columns != "G")].div(summed_basic_stats["G"],
                                                                                                axis=0)
avg_stats = avg_stats.apply(custom_round_up, args=[1])

# bring the stats back to a 36 mn based
avg_stats_36_minutes = avg_stats.div((avg_stats["MP"] / 36), axis=0)
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[1])
names = pd.DataFrame(avg_stats_36_minutes.index)

avg_stats_36_minutes.to_csv("../csv/avg_stats_36_minutes_unscaled.csv")

# Scaling
avg_stats_36_minutes = avg_stats_36_minutes - avg_stats_36_minutes.min()
avg_stats_36_minutes = avg_stats_36_minutes / (avg_stats_36_minutes.max() - avg_stats_36_minutes.min())
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[2])
avg_stats_36_minutes_scaled = avg_stats_36_minutes.drop(columns=["MP"])


# retrieving advanced stats
ad_2016 = pd.read_csv(csv_files_location + 'NBA_advanced_2015-2016.csv')
ad_2017 = pd.read_csv(csv_files_location + 'NBA_advanced_2016-2017.csv')
ad_2018 = pd.read_csv(csv_files_location + 'NBA_advanced_2017-2018.csv')
ad_2019 = pd.read_csv(csv_files_location + 'NBA_advanced_2018-2019.csv')
ad_2020 = pd.read_csv(csv_files_location + 'NBA_advanced_2019-2020.csv')

# remove special characters, accent, JR, SR, etc...
ad_2016 = clean_names(ad_2016, "Player")
ad_2017 = clean_names(ad_2017, "Player")
ad_2018 = clean_names(ad_2018, "Player")
ad_2019 = clean_names(ad_2019, "Player")
ad_2020 = clean_names(ad_2020, "Player")

# remving TOT rows for players whi have been traded during the season
ad_2016 = ad_2016[ad_2016["Tm"] != "TOT"]
ad_2017 = ad_2017[ad_2017["Tm"] != "TOT"]
ad_2018 = ad_2018[ad_2018["Tm"] != "TOT"]
ad_2019 = ad_2019[ad_2019["Tm"] != "TOT"]
ad_2020 = ad_2020[ad_2020["Tm"] != "TOT"]

# let's just keep the cols we are interested in
ad_2016 = ad_2016.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2017 = ad_2017.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2018 = ad_2018.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2019 = ad_2019.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2020 = ad_2020.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]


# for the advanced stats we need to ponderate stats by the nb of games played during the season
def ponderateByGamesPlayed(df):
    # let's get the name, minute played and nb of games played
    names = df["Player"]
    minutes = df["MP"]
    games = df["G"]

    # let's remove minutes played, name and nb of game played
    df = df.drop(columns=["Player", "MP", "G"])

    # multiply the stats by the nb of games played during the season
    df = df.mul(games, axis=0)

    # let's add minutes played, name and nb of game played
    res = pd.concat([names, games, minutes, df], axis=1)

    # ad cols names
    res.columns = ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]
    return res


# apply the above function for every year
ad_2016 = ponderateByGamesPlayed(ad_2016)
ad_2017 = ponderateByGamesPlayed(ad_2017)
ad_2018 = ponderateByGamesPlayed(ad_2018)
ad_2019 = ponderateByGamesPlayed(ad_2019)
ad_2020 = ponderateByGamesPlayed(ad_2020)

# concat row wise the stats on the last 5 years
summed_ad = ad_2016.append(ad_2017).append(ad_2018).append(ad_2019).append(ad_2020)

# defining aggregation function for every column
agr = {'MP': ['sum'], 'G': ['sum'], 'PER': ['sum'], 'TS%': ['sum'], '3PAr': ['sum'], 'TRB%': ['sum'], 'USG%': ['sum'],
       'OWS': ['sum'], 'DWS': ['sum']}
agg_advanced = summed_ad.groupby(['Player']).agg(agr)


# remove those did not play enough minute or games
agg_advanced = agg_advanced.loc[(agg_advanced['G']["sum"] > 30) & (agg_advanced['MP']["sum"] > 500)]

# lets retrieve the players height
heights = pd.read_csv(csv_files_location + "players_height.csv")
heights = clean_names(heights, "Name")
heights = heights[["Name", "Height (cm)"]]
heights = heights.rename(columns={"Name": "Player"})

# bring the stats on a per game-based
games = agg_advanced["G"]["sum"]
final_advanced = agg_advanced.div(games, axis=0)
final_advanced = final_advanced.drop(columns=["G"])
final_advanced = final_advanced.apply(custom_round_up, args=[2])
final_advanced = pd.concat([games, final_advanced], axis=1)

# we add the players height
final_advanced = pd.merge(final_advanced, heights, on="Player")
final_advanced = final_advanced.set_index("Player")

# save to csv
final_advanced.columns = ["G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS", "Height"]
final_advanced.to_csv("../csv/unscaled_aggregated_stats.csv")

# Scaling
final_advanced_scaled = final_advanced - final_advanced.min()
final_advanced_scaled = final_advanced_scaled / (final_advanced_scaled.max() - final_advanced_scaled.min())

# merging advanced stats, basic stats and players name
final = pd.merge(final_advanced_scaled, avg_stats_36_minutes_scaled, on="Player")

# by merging, we remove the retired players
final = pd.merge(team_and_player, final, on="Player")

# export to csv
final.to_csv("../csv/players_stats.csv")
