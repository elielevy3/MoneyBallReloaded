#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:14:15 2021

@author: elie
"""
from nbformat import write
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import streamlit as st


# list of players with all the stats we have
stats = pd.read_csv("csv/players_stats.csv")


def find_most_similar_player_by_criterias(player_stats, nb_most_similar, criterias):
    player_names = stats["Player"]

    # we keep the interesting value
    df = stats[criterias]

    # number of player in the df
    nb_of_players = len(df.index)

    # our distance matrix
    dist_mat_dict = {}
    
    # st.write(df)
    # st.write(player_stats)

    # st.write(type(df))
    # st.write(type(player_stats))

    if type(player_stats) != list: 
        player_stats = player_stats.iloc[0] 

    for i in range(nb_of_players):
        dist = round(distance.euclidean(df.iloc[i], player_stats), 3)
        dist_mat_dict[player_names[i]] = dist

    # sorting the distance (ascending order)
    dist_mat_dict = {k: v for k, v in sorted(dist_mat_dict.items(), key=lambda item: item[1])}

    # get max and min to scale
    nb_most_similar_values = list(dist_mat_dict.values())
    min_value = min(nb_most_similar_values)
    max_value = max(nb_most_similar_values)

    # just keep the nth most similar
    dist_mat_dict = {k: v for k, v in list(dist_mat_dict.items())[1:nb_most_similar+1]}

    # scaling + project onto a 0-100 score
    dist_mat_dict = {k: round(abs(1-(v - min_value)/(max_value-min_value)) * 100, 2) for k, v in dist_mat_dict.items()}

    return dist_mat_dict.items()


def computing_distance_matrix(source, criterias):
    player_names = source["Player"]
    
    # we keep the interesting value
    df = source[criterias]
    
    # number of player
    nb_of_players = len(df.index)
    
    # our distance matrix
    dist_mat_dict = {}
    
    # let's compute the distance for every couple of players
    for i in range(nb_of_players):
        dist_mat_dict[player_names[i]] = {}
        for j in range(nb_of_players):
            dist_mat_dict[player_names[i]][player_names[j]] = round(distance.euclidean(df.iloc[i], df.iloc[j]), 3)

    # list is more convenient for scaling
    # here we have a list of lists
    distance_matrix_list = [list(z.values()) for y,z in dist_mat_dict.items()]
    distance_matrix_list = pd.DataFrame(distance_matrix_list)
    min_of_distance = distance_matrix_list.min().min()
    max_of_distance = distance_matrix_list.max().max()

    # we fill back the value from the list to the dict
    for i in range(nb_of_players):
        for j in range(nb_of_players):
            
            # scaling before
            distance_matrix_list[i][j] = distance_matrix_list[i][j] - min_of_distance
            distance_matrix_list[i][j] = distance_matrix_list[i][j] / (max_of_distance - min_of_distance)
            
            # in order to have a 0-100% confidence index
            # let's do the 1 complement value and multiple by 100
            # with two digits after the coma
            val = round(abs(1 - distance_matrix_list[i][j])*100, 2)
            dist_mat_dict[player_names[i]][player_names[j]] = val
    
    # lets save it so we do not have to compute everytime
    dist_mat_df = pd.DataFrame(dist_mat_dict)
    dist_mat_df.to_csv("../csv/distance_matrix.csv")
    # return dist_mat_dict


# return a dict of dict
def get_distance_between_players(list_of_players, dist_matrix):
    # lets sort it to have the same order on both axis
    list_of_players = sorted(list_of_players)
    dist_mat_dict = {}
    for player in list_of_players:
        dist_mat_dict[player] = {}
        for player2 in list_of_players:
            dist_mat_dict[player][player2] = dist_matrix[dist_matrix["Name"] == player].iloc[0][player2]
            
    return dist_mat_dict


def get_distance_between_players_with_criterias(list_of_players, criterias):
    # we add player name
    enlarged_criterias = [value for value in criterias]
    enlarged_criterias.append("Player")
    enlarged_criterias_df = stats[enlarged_criterias]
    cross_player_dict = {}

    for player in list_of_players:
        dict_of_dist_per_player = {}
        # for every player, we compute distance with every other players to get the mix and max in order to normalize
        # for this given player
        player_stat = enlarged_criterias_df[enlarged_criterias_df["Player"] == player][criterias]
        
        if type(player_stat) != list: 
            player_stat = player_stat.iloc[0] 

        for row in enlarged_criterias_df.itertuples():
            # remove the first column (the index) and the last one (name)
            row_stats = row[1:-1]
            dist = round(distance.euclidean(player_stat, row_stats), 3)
            dict_of_dist_per_player[row.Player] = dist

        # get the min and max
        list_of_dist_for_one_player = list(dict_of_dist_per_player.values())
        min_value = min(list_of_dist_for_one_player)
        max_value = max(list_of_dist_for_one_player)

        cross_player_dict[player] = {}
        # normalize distance row-wise (for one given player)
        for player_2 in list_of_players:
            normalized_value = round(abs(1-(dict_of_dist_per_player[player_2]-min_value)/(max_value-min_value))*100, 2)
            cross_player_dict[player][player_2] = normalized_value

    return cross_player_dict


# return a list of 2-elements tuples (name, similarity score)
def get_most_similar_players(player_name, nb_of_similar_players_wanted, dist_mat):
    
    #lets sort the list of similarity between player and the rest of the NBA
    sorted_similarity = dict(sorted(dist_mat[player_name].items(), key=lambda item: item[1], reverse=True))

    #lets keep the n first (Except the the closest who is the player himself)
    most_similar_players = list(sorted_similarity.items())[1:nb_of_similar_players_wanted+1]
    
    # retrieve the players name instead of his index number
    for i in range(len(most_similar_players)):
        index_value = most_similar_players[i][0]
        name = dist_mat["Name"][index_value]
        similarity_confidence = most_similar_players[i][1]
        most_similar_players[i] = (name, similarity_confidence)
                
    return most_similar_players


def plot_heat_matrix(only_number_matrix, list_of_players):
    list_of_players = sorted(list_of_players)
    #lets try to plot a heat matrix
    fig = plt.figure()
    c = plt.imshow(only_number_matrix, cmap='Reds', interpolation='nearest')
    plt.title("Similarity of players")
    plt.colorbar(c)
    # rotate to prevent players name from overlapping
    plt.xticks(np.arange(0, len(list_of_players)) , list_of_players, rotation=270)
    plt.yticks(np.arange(0, len(list_of_players)) , list_of_players)
    plt.show()
    return fig

