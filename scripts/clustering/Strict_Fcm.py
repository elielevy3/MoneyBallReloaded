#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sys
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans

sys.path.append("/home/elie/Documents/MoneyballReloaded/scripts")
from Polygone import performance_polygon_vs_player

df = pd.read_csv('/home/elie/Documents/MoneyballReloaded/csv/players_stats.csv')
player_names = df["Player"]

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])

# we keep the interesting value
df_fcm = df[['Player', 'TRB', 'PTS', 'AST', "DWS", '3PA', "OWS", "USG%", "Height"]]

# we keep the players name for later
players_name = df_fcm["Player"]

# we remove the player column for the computation
df_fcm = df_fcm.loc[:,(df_fcm.columns != "Player")]

# Computation
nb_cluster_fuzzy = 35
fuzzy_kmeans = FuzzyKMeans(k=nb_cluster_fuzzy, m=1.1)
fuzzy_kmeans.fit(df_fcm)
fuzzy_clusters = pd.DataFrame(fuzzy_kmeans.fuzzy_labels_)

# we add the players name back
fuzzy_clusters = pd.concat([players_name, fuzzy_clusters], axis=1)

nb_max_players_per_cluster_fcm = 3

final_clusters = pd.DataFrame()


for i in range(nb_cluster_fuzzy):
    # lets keep the coresponding col of membership degree
    sets = fuzzy_clusters[["Player", i]]
    
    # lets sort
    sets = sets.sort_values(by=i, ascending=False)
    
    #let's juste keep the top n% and be sure they are above a threeshold
    sets = sets.head(nb_max_players_per_cluster_fcm)
    sets = sets[["Player"]]
    print("we add "+str(nb_max_players_per_cluster_fcm)+" clustered players")
    
    # remove the hard clustered players from the fuzzy df to avoid having duplicates
    fuzzy_clusters = fuzzy_clusters[~fuzzy_clusters['Player'].isin(list(sets["Player"]))]
    
    #lets add the # of the cluster
    sets["Cluster"] = i+1
    #add those lines to the previous results
    final_clusters = pd.concat([final_clusters, sets], axis=0)


#now let's print the overlapping polygones for each cluster
for i in final_clusters.Cluster.unique():   
    players_to_draw = final_clusters[final_clusters["Cluster"] == i]["Player"].tolist()
    properties = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]
    performance_polygon_vs_player(players_to_draw, properties)


print("Now we have "+str(len(final_clusters.index))+" players clustered out of "+str(len(clustering_df.index))+" players.")
