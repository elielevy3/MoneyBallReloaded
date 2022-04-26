#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:20:27 2021

@author: elie
""" 
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd
import numpy as np
from Polygone import performance_polygon_vs_player
from Dissimilarity_Matrix import get_most_similar_players, plot_heat_matrix, get_distance_between_players

stats = pd.read_csv("../csv/players_stats.csv")
dist_mat = pd.read_csv("../csv/distance_matrix.csv")
dist_mat = dist_mat.rename(columns={"Unnamed: 0": 'Name'})
criterias = ['OWS', 'DWS', 'AST', 'TS%', "TRB", "PTS", "3PA", "BLK"]
salaries = pd.read_csv("../csv/players_salaries.csv")
salaries.set_index("Unnamed: 0", inplace=True)
potential_criterias = ['MP', 'TS%', '3PAr', 'TRB%', 'USG%', 'OWS', 'DWS', 'Height', 'FGA', '3P', '3PA', '2P',
                       '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

st.title('MoneyBall Reloaded')

st.header("Pick the criterias that matters the most for you")
picked_criterias_dict = {}
picked_criterias_array = []
i = 0
number_of_item_per_line = 6
col_size = np.ones(number_of_item_per_line)
while i < len(potential_criterias):
    cols = st.beta_columns(col_size)
    for index, col in enumerate(potential_criterias[i:i + number_of_item_per_line]):
        with cols[index]:
            picked_criterias_dict[col] = st.checkbox(col)
    i = i + number_of_item_per_line

for key, value in picked_criterias_dict.items():
    if value:
        picked_criterias_array.append(key)

# st.text("A doubt on what those criterias mean ? ")
st.write("A doubt on what those criterias mean ? [Check this out](https://www.basketball-reference.com/about/glossary.html)")
st.markdown("____")

st.header("Find the most similar players to the player of your choosing")
st.text("")

player = st.selectbox(
    'Which player do you want to find similar players to?', stats['Player'])

number = st.selectbox(
    'How many players do you want among the most similar?', np.arange(1, 10, 1))


# get the n most similar to the required player
most_similar_players = get_most_similar_players(player, number, dist_mat)
most_similar_players_names = [names for (names, score) in most_similar_players ]
most_similar_players_names.append(player)


# get age of n most similar player
players_age = pd.read_csv("../csv/NBA_totals_2019-2020.csv")[["Player", "Age"]]
players_age.columns = ["Name", "Age"]

# get the distance between the n most similar players of the required player
players_distances = get_distance_between_players(most_similar_players_names, dist_mat)
only_number_matrix = [list(value.values()) for key, value in players_distances.items()]

# transform to proper df
df_most_similar_players = pd.DataFrame(most_similar_players)
df_most_similar_players.columns = ["Name", "Similarity"]
df_most_similar_players = pd.merge(df_most_similar_players, players_age, on="Name")
df_most_similar_players = pd.merge(df_most_similar_players, salaries, on="Name")
df_most_similar_players.columns = ["Name", "Similarity", "Age", "Salary"]
# df_most_similar_players.set_index('Name', inplace=True)

# get the heat matrix of the n most similar players
heat_matrix = plot_heat_matrix(only_number_matrix, most_similar_players_names)

# draw polygones for the n most similar players
players_to_draw = [player[0] for player in most_similar_players]
players_to_draw.append(player)
polygones = performance_polygon_vs_player(players_to_draw, picked_criterias_array)

# heat_matrix.savefig("heat_matrix.jpg")
# polygones.savefig("polygones.jpg")


# Display
st.table(df_most_similar_players)
st.write(heat_matrix)
st.write(polygones)

st.markdown("____")


# heat_matrix_image = mpimg.imread('heat_matrix.jpg')
# polygones_image = mpimg.imread("polygones.jpg")

# col1, col2 = st.beta_columns(2)

# col1.header("Polygones")
# col1.image(polygones_image, use_column_width=True)

# col2.header("Heat Matrix")
# col2.image(heat_matrix_image, use_column_width=True)

st.header("Compare the players you want together")

nb_of_player_to_compare = st.selectbox(
    'How many players do you want to individually compare to each other?',
     np.arange(1, 10, 1))

players_list = []

for i in range(nb_of_player_to_compare):
    player_picked = st.selectbox(
    'Which player do you want to find similar players to?',
     stats['Player'], key=i)
    players_list.append(player_picked)
    

# get the distance between the players selected
players_distances = get_distance_between_players(players_list, dist_mat)
only_number_matrix = [list(value.values()) for key, value in players_distances.items()]

# get the heat matrix of the selected players
heat_matrix = plot_heat_matrix(only_number_matrix, players_list)

# draw polygones for the selected players
polygones = performance_polygon_vs_player(players_list, picked_criterias_array)

# heat_matrix.savefig("heat_matrix.jpg")
# polygones.savefig("polygones.jpg")


# Display
st.write(polygones)
st.write(heat_matrix)

