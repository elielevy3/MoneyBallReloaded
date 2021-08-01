#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:14:15 2021

@author: elie
"""
import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import distance
from Polygone import performance_polygon_vs_player

# to check how dit it take
start_time = time.time()

def computing_distance_matrix(source, criterias):    
    player_names = source["Player"]
    
    # we keep the interesting value
    df = source[criterias]
    
    # number of player
    nb_of_players = len(df.index)
    
    # our distance matrix
    dist_mat_dict = {}
    
    #lets compute the distance for every couple of players
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

    #return dist_mat_dict

# return a dict of dict
def get_distance_between_players(list_of_players, dist_matrix):
    #lets sort it to have the same order on both axis
    list_of_players = sorted(list_of_players)
    dist_mat_dict = {}
    for player in list_of_players:
        dist_mat_dict[player] = {}
        for player2 in list_of_players:
            dist_mat_dict[player][player2] = dist_matrix[dist_matrix["Name"] == player].iloc[0][player2]
            
    return dist_mat_dict

#return a list of 2-elements tuples (name, similarity score)
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
        most_similar_players[i] = (name,similarity_confidence)
                
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
    
    
source = pd.read_csv('../csv/players_stats.csv')
criterias = ['TRB', 'PTS', 'AST', 'DWS', '3PA', "OWS", "USG%", "Height"]

#retrieving the data
dist_mat = pd.read_csv("../csv/distance_matrix.csv")        
dist_mat = dist_mat.rename(columns={"Unnamed: 0": 'Name'})

# get the n most similar player to X and get the similarity values between each and every one of them
player = "Bradley Beal"
most_similar_players = get_most_similar_players(player, 4, dist_mat)
most_similar_players_names = [names for (names, score) in most_similar_players ]

players_distances = get_distance_between_players(most_similar_players_names, dist_mat)
only_number_matrix = [list(value.values()) for key, value in players_distances.items()]

# plot the heat matrix of several players
plot_heat_matrix(only_number_matrix, most_similar_players_names)

# draw polygones
players_to_draw = [player[0] for player in most_similar_players]
players_to_draw.append(player)
properties = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]
performance_polygon_vs_player(players_to_draw, properties)
    

# ignore warnings for the polygone display
warnings.filterwarnings("ignore")

print("--- %s seconds ---" % round(time.time() - start_time, 2))
