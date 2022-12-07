#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:20:27 2021

@author: elie
"""
import time
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for working with sample data.
import pandas as pd
import numpy as np
import Unidecode
from Polygone import performance_polygon_vs_player
from Dissimilarity_Matrix import find_most_similar_player_by_criterias, get_distance_between_players_with_criterias

st.set_page_config(layout="wide", page_icon="🏀", page_title="MoneyBall Reloaded")

def clean_names(df, col_name):
    df[col_name] = df[col_name].apply(str.replace, args=[" Jr.", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" Sr.", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" III", ""])
    df[col_name] = df[col_name].apply(str.replace, args=[" II", ""])
    df[col_name] = df[col_name].apply(Unidecode.unidecode)
    df[col_name] = df[col_name] = df[col_name].apply(str.replace, args=[".", ""])
    return df

@st.experimental_memo
def get_player_stats():
    return pd.read_csv("../csv/players_stats.csv")

@st.experimental_memo
def get_player_salaries():
    return pd.read_csv("../csv/players_salaries.csv")


stats = get_player_stats()
initial_criterias = ['OWS', 'DWS', 'AST', 'TS%', "TRB", "PTS", "3PA", "BLK"]
salaries = get_player_salaries()
salaries.set_index("Unnamed: 0", inplace=True)
potential_criterias = ['MP', 'TS%', '3PAr', 'TRB%', 'USG%', 'OWS', 'DWS', 'Height', 'FGA', '3P', '3PA', '2P',
                       '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

basic_criterias = ["MP", "FGA", '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB',
                   'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# make sure every player is in salaries
salaries.columns = ["Player", "Salaries"]
salaries.set_index("Player", inplace=True)
stats = stats.join(salaries, how="left", on="Player")

# get age of players
players_age = pd.read_csv("../csv/NBA_totals_2019-2020.csv")[["Player", "Age"]]
players_age = clean_names(players_age, "Player")
players_age = players_age.drop_duplicates()

# get avg PER of every player on the last years
players_per = pd.read_csv("../csv/unscaled_aggregated_stats.csv")
players_per = players_per[["Player", "PER"]]

# displaying
st.title('MoneyBall Reloaded 🏀')
# st.header("Pick the criterias that matters the most for you")
# picked_criterias_dict = {}
# picked_criterias_array = []
# i = 0
# number_of_item_per_line = 6
# col_size = np.ones(number_of_item_per_line)
# while i < len(potential_criterias):
#     cols = st.columns(col_size)
#     for index, col in enumerate(potential_criterias[i:i + number_of_item_per_line]):
#         with cols[index]:
#             picked_criterias_dict[col] = st.checkbox(col)
#     i = i + number_of_item_per_line


picked_criterias_array = st.sidebar.multiselect("Pick the criterias", potential_criterias)

# picked_criterias_array = [key for key, value in picked_criterias_dict.items() if value]

if len(picked_criterias_array) == 0:
    picked_criterias_array.append("PTS")

st.write(
    "A doubt on what those criterias mean ? [Check this out](https://www.basketball-reference.com/about/glossary.html)")

st.write("Want to check out the github repo ? [Help yourself](https://github.com/elielevy3/MoneyBallReloaded)")
st.markdown("____")

#c1, c3, c2 = st.tabs(["Find the most similar players to a player", "Describe the player you need", "Compare the players you want together"])

c1, _, c3, _, c2 = st.columns([8, 1,  8, 1, 8])

polygone_col_1, _, polygone_col_2, _, polygone_col_3 = st.columns([8, 1,  8, 1, 8])

c1.header("Find the most similar players to a given one")
c1.text("")

player = c1.selectbox(
    'Which player do you want to find similar players to?', stats['Player'])

number = c1.selectbox(
    'How many players do you want among the most similar?', np.arange(1, 10, 1))

# get the n most similar to the required player
player_to_compute = stats[stats["Player"] == player][picked_criterias_array]
most_similar_players = find_most_similar_player_by_criterias(player_to_compute, number, picked_criterias_array)


# transform to proper df
df_most_similar_players = pd.DataFrame(most_similar_players)

df_most_similar_players.columns = ["Player", "Similarity"]

df_most_similar_players = pd.merge(df_most_similar_players, players_age, on="Player")

df_most_similar_players = pd.merge(df_most_similar_players, stats[["Player", "Salaries"]], on="Player")

df_most_similar_players = pd.merge(df_most_similar_players, players_per, on="Player")

df_most_similar_players.columns = ["Player", "Similarity", "Age", "Salary", "PER"]

# draw polygones for the n most similar players
players_to_draw = [player[0] for player in most_similar_players]
players_to_draw.append(player)
if len(picked_criterias_array) == 0:
    picked_criterias_array.append("PTS")

polygones = performance_polygon_vs_player(players_to_draw, picked_criterias_array)

df_most_similar_players.index = df_most_similar_players.index + 1

# Display
c1.table(df_most_similar_players[["Player", "Similarity", "Age", "Salary", "PER"]])

#c1.write(polygones)
polygone_col_1.write(polygones)

#st.markdown("____")

c2.header("Compare the players you want together")
c2.text("")

nb_of_player_to_compare = c2.selectbox(
    'How many players do you want to individually compare to each other?',
    np.arange(1, 10, 1))

players_list = []

for i in range(nb_of_player_to_compare):
    player_picked = c2.selectbox(
        'Which player do you want to find similar players to?',
        stats['Player'], key=i)
    players_list.append(player_picked)

# get the distance between the players selected
players_distances = get_distance_between_players_with_criterias(players_list, picked_criterias_array)


c2.markdown("#")
c2.markdown("#")
c2.markdown("#")

# draw polygones for the selected players
polygones = performance_polygon_vs_player(players_list, picked_criterias_array)


df_most_similar_players = pd.DataFrame(players_list)

df_most_similar_players.columns = ["Player"]

df_most_similar_players = pd.merge(df_most_similar_players, players_age, on="Player")

df_most_similar_players = pd.merge(df_most_similar_players, stats[["Player", "Salaries"]], on="Player")

df_most_similar_players = pd.merge(df_most_similar_players, players_per, on="Player")

df_most_similar_players.columns = ["Player", "Age", "Salary", "PER"]

c2.write(df_most_similar_players)
#c2.write(polygones)
polygone_col_3.write(polygones)
#c3, c4 = st.columns((2, 3))

# Describe the player you would like
c3.header("Describe the player you need")
nb_of_player_to_compare_to_fictive = c3.selectbox('How many similar players do you want ? ', np.arange(1, 10, 1))
c3.write("------")

fictive_player_criterias_dict = {}

# intersect picked criterias for fictive player with basic criterias because users will not fill advanced stats
fictive_player_criterias_array = [elem for elem in set(picked_criterias_array) if
                                  elem in set(basic_criterias)]

#fictive_player_criterias_array = picked_criterias_array
fictive_player_criterias_array.append("MP")

# declaring variables to display the selected criterias
i = 0
number_of_item_per_line_fictive_player = 10
col_size_fictive_player = np.ones(number_of_item_per_line_fictive_player)

fictive_player_criterias_array = list(set(fictive_player_criterias_array))

# for every criterias
while i < len(fictive_player_criterias_array):
    cols_fictive_player = st.columns(col_size_fictive_player)
    for index, col in enumerate(fictive_player_criterias_array[i:i + number_of_item_per_line_fictive_player]):
        with cols_fictive_player[index]:
            fictive_player_criterias_dict[col] = c3.number_input(col, step=1)
    i = i + number_of_item_per_line_fictive_player


c3.markdown("____")

# creating df with just one line for the fictive player
fictive_player_stats_array = [v for v in fictive_player_criterias_dict.values()]
fictive_player_stats_pd = pd.DataFrame([fictive_player_stats_array])
fictive_player_stats_pd.columns = fictive_player_criterias_array

# now we need to min max scale this new fictive player for all of his feature
unscaled_advanced_data = pd.read_csv("../csv/unscaled_aggregated_stats.csv")
unscaled_basic_data = pd.read_csv("../csv/avg_stats_36_minutes_unscaled.csv")

fictive_player_stats_array = []

# we dont want to scale the nb of minute
fictive_player_criterias_array.remove("MP")

# scaling the new fictive player
if len(fictive_player_criterias_array) != 0:
    unscaled_basic_data = unscaled_basic_data[fictive_player_criterias_array]
    
    for col in fictive_player_criterias_array:
        val = int(fictive_player_stats_pd[col].tolist()[0])
        min_value = unscaled_basic_data[col].min()
        max_value = unscaled_basic_data[col].max()
        fictive_player_stats_array.append((val - min_value) / (max_value - min_value))

    similar_players_to_fictive = find_most_similar_player_by_criterias(fictive_player_stats_array, nb_of_player_to_compare_to_fictive, fictive_player_criterias_array)

    # transform to proper df
    df_most_similar_players = pd.DataFrame(similar_players_to_fictive)

    df_most_similar_players.columns = ["Player", "Similarity"]

    df_most_similar_players = pd.merge(df_most_similar_players, players_age, on="Player")

    df_most_similar_players = pd.merge(df_most_similar_players, stats[["Player", "Salaries"]], on="Player")

    df_most_similar_players = pd.merge(df_most_similar_players, players_per, on="Player")

    df_most_similar_players.columns = ["Player", "Similarity", "Age", "Salary", "PER"]

    # draw polygones for the n most similar players
    players_to_draw = [player[0] for player in similar_players_to_fictive]
    # players_to_draw.append(player)
    if len(picked_criterias_array) == 0:
        picked_criterias_array.append("PTS")

    polygones = performance_polygon_vs_player(players_to_draw, picked_criterias_array)

    df_most_similar_players.index = df_most_similar_players.index + 1

    # Display
    c3.table(df_most_similar_players[["Player", "Similarity", "Age", "Salary", "PER"]])
    #c3.write(polygones)
    polygone_col_2.write(polygones)

    #c3.markdown("____")
