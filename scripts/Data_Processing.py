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

# on recupere les stats de base
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

# on recupère l'équipe finale de chaque joueur de cette année
# on fait d'une pierre deux coups en récuperant les noms et en filtrant les joueurs retraités
# avant la derniere saison
team_and_player = df_2020[["Player", "Tm", 'Pos']]
team_and_player["final_team"] = team_and_player.groupby('Player')['Tm'].transform('last')
team_and_player = team_and_player[["Player", "final_team", "Pos"]]
team_and_player = team_and_player.drop_duplicates(subset=['Player'])

# on enleve les lignes TOT pour les joueurs qui ont été tranféré en cours de saison:
df_2016 = df_2016[df_2016["Tm"] != "TOT"]
df_2017 = df_2017[df_2017["Tm"] != "TOT"]
df_2018 = df_2018[df_2018["Tm"] != "TOT"]
df_2019 = df_2019[df_2019["Tm"] != "TOT"]
df_2020 = df_2020[df_2020["Tm"] != "TOT"]

# on ne garde que les colonnes qui nous interessent
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

# on concatene en hauteur tous les df
basic_stats = basic_stats_2016.append(basic_stats_2017).append(basic_stats_2018).append(basic_stats_2019).append(
    basic_stats_2020)

# on group par joueur
summed_basic_stats = basic_stats.groupby(['Player']).sum()

# on enleve ceux qui ont joué moins de 30 matches ou 1000 Minutes
# summed_basic_stats = summed_basic_stats.loc[(summed_basic_stats['G'] > 100) | (summed_basic_stats['MP'] > 2500)]


# on arrondi a un chiffre après la virgule
def custom_round_up(x, y):
    return round(x, y)


avg_stats = summed_basic_stats.loc[:,
            (summed_basic_stats.columns != "Player") & (summed_basic_stats.columns != "G")].div(summed_basic_stats["G"],
                                                                                                axis=0)
avg_stats = avg_stats.apply(custom_round_up, args=[1])

# on doit ramener sur 36 minutes
avg_stats_36_minutes = avg_stats.div((avg_stats["MP"] / 36), axis=0)
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[1])
names = pd.DataFrame(avg_stats_36_minutes.index)

# Scaling
avg_stats_36_minutes = avg_stats_36_minutes - avg_stats_36_minutes.min()
avg_stats_36_minutes = avg_stats_36_minutes / (avg_stats_36_minutes.max() - avg_stats_36_minutes.min())
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[2])
avg_stats_36_minutes_scaled = avg_stats_36_minutes.drop(columns=["MP"])

# on recupere les stats avancées
ad_2016 = pd.read_csv(csv_files_location + 'NBA_advanced_2015-2016.csv')
ad_2017 = pd.read_csv(csv_files_location + 'NBA_advanced_2016-2017.csv')
ad_2018 = pd.read_csv(csv_files_location + 'NBA_advanced_2017-2018.csv')
ad_2019 = pd.read_csv(csv_files_location + 'NBA_advanced_2018-2019.csv')
ad_2020 = pd.read_csv(csv_files_location + 'NBA_advanced_2019-2020.csv')

# on enleve les accents et caractères spéciaux du nom des joueurs pour les grouper
ad_2016 = clean_names(ad_2016, "Player")
ad_2017 = clean_names(ad_2017, "Player")
ad_2018 = clean_names(ad_2018, "Player")
ad_2019 = clean_names(ad_2019, "Player")
ad_2020 = clean_names(ad_2020, "Player")

# on enleve les lignes TOT pour les joueurs qui ont été tranféré en cours de saison: 
ad_2016 = ad_2016[ad_2016["Tm"] != "TOT"]
ad_2017 = ad_2017[ad_2017["Tm"] != "TOT"]
ad_2018 = ad_2018[ad_2018["Tm"] != "TOT"]
ad_2019 = ad_2019[ad_2019["Tm"] != "TOT"]
ad_2020 = ad_2020[ad_2020["Tm"] != "TOT"]

# on ne garde que les colonnes qui nous intéresse
ad_2016 = ad_2016.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2017 = ad_2017.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2018 = ad_2018.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2019 = ad_2019.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]
ad_2020 = ad_2020.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]]


# pour les stats avancées on a besoin de pondérer les stats d'une saison par le nb de matches joués
def ponderateByGamesPlayed(df):
    # On recupere les noms, minutes jouées et matches joués
    names = df["Player"]
    minutes = df["MP"]
    games = df["G"]

    # on enleve les noms, minutes jouées et matches joués
    df = df.drop(columns=["Player", "MP", "G"])

    # on multiplie chaque stats de chaque joueur par le nb de matches joués pendant cette saison
    df = df.mul(games, axis=0)

    # on rajoute les noms, les minutes et des matches joués
    res = pd.concat([names, games, minutes, df], axis=1)

    # on rajoute le nom des colonnes
    res.columns = ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]
    return res


# on applique la fonction pour pondérer par le nb de match joué
ad_2016 = ponderateByGamesPlayed(ad_2016)
ad_2017 = ponderateByGamesPlayed(ad_2017)
ad_2018 = ponderateByGamesPlayed(ad_2018)
ad_2019 = ponderateByGamesPlayed(ad_2019)
ad_2020 = ponderateByGamesPlayed(ad_2020)

# on concat les stats sur les 5 dernieres saisons avant de les aggréger par joueur
summed_ad = ad_2016.append(ad_2017).append(ad_2018).append(ad_2019).append(ad_2020)

# On agrege
agr = {'MP': ['sum'], 'G': ['sum'], 'PER': ['sum'], 'TS%': ['sum'], '3PAr': ['sum'], 'TRB%': ['sum'], 'USG%': ['sum'],
       'OWS': ['sum'], 'DWS': ['sum']}
agg_advanced = summed_ad.groupby(['Player']).agg(agr)

# on enleve ceux qui ont joué moins de 100 matches => c'est pour cela qu'on a moins de joueurs à la fin!!!!
# agg_advanced = agg_advanced.loc[(agg_advanced["MP"]["sum"] > 25) | (agg_advanced["G"]["sum"] > 1)]

# agg_advanced = agg_advanced.loc[(agg_advanced['G']["sum"] > 100) & (agg_advanced['MP']["sum"] > 2500) ]

# lets retrieve the players height
heights = pd.read_csv(csv_files_location + "players_height.csv")
heights = clean_names(heights, "Name")
heights = heights[["Name", "Height (cm)"]]
heights = heights.rename(columns={"Name": "Player"})

# on ramene les stats par matches
games = agg_advanced["G"]["sum"]
final_advanced = agg_advanced.div((games), axis=0)
final_advanced = final_advanced.drop(columns=["G"])
final_advanced = final_advanced.apply(custom_round_up, args=[2])
final_advanced = pd.concat([games, final_advanced], axis=1)

# we add the players height
final_advanced = pd.merge(final_advanced, heights, on="Player")
final_advanced = final_advanced.set_index("Player")

final_advanced.columns = ["G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS", "Height"]

# Scaling
final_advanced_scaled = final_advanced - final_advanced.min()
final_advanced_scaled = final_advanced_scaled / (final_advanced_scaled.max() - final_advanced_scaled.min())

# on fusionne les stats avancées, les stats de base et les noms des joueurs
final = pd.merge(final_advanced_scaled, avg_stats_36_minutes_scaled, on="Player")

# by merging, we remove the retired players
final = pd.merge(team_and_player, final, on="Player")

# export to csv
final.to_csv("../csv/players_stats.csv")
