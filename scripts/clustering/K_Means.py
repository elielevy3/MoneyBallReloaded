#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:00:51 2021

@author: elie
"""

import pandas as pd
import numpy as np
import warnings
import sklearn
import sys
import time
from sklearn.cluster import KMeans
sys.path.append("/home/elie/Documents/MoneyballReloaded/scripts")
from Polygone import performance_polygon_vs_player

start_time = time.time()

# get the data
source = pd.read_csv('/home/elie/Documents/MoneyballReloaded/csv/players_stats.csv')
df_to_compute = source.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])

results = pd.DataFrame(data = None, columns = ['score'],dtype=np.float64) 

# computing the optimal number of cluster 
min_score_davies = 100000

max_score_silhouette = -100000

for nb_cluster_test in np.arange(start = 3,stop=100): 
            kmeans = KMeans(n_clusters=nb_cluster_test, random_state=0).fit(df_to_compute)
            score_davies = sklearn.metrics.davies_bouldin_score(df_to_compute,kmeans.labels_)
            score_silhouette = sklearn.metrics.silhouette_score(df_to_compute,kmeans.labels_)

            if (score_davies < min_score_davies):
                min_score_davies = score_davies
                optimal_number_of_cluster_davies = nb_cluster_test
            if (score_silhouette > max_score_silhouette):
                max_score_silhouette = score_silhouette
                optimal_number_of_cluster_silhouette = nb_cluster_test
                
optimal_number_of_cluster = int((optimal_number_of_cluster_davies + optimal_number_of_cluster_silhouette) / 2)

# compute K-MEANS
#nb_clusters = 70
kmeans = KMeans(n_clusters=optimal_number_of_cluster, random_state=0).fit(df_to_compute)

# average number of player per cluster
avg_number_per_cluster = round(len(df_to_compute.index) / optimal_number_of_cluster, 2)

# get the clusters
clusters = pd.DataFrame(kmeans.labels_)
clusters.columns = {"Cluster"}

# stick the cluster number for each player
clustered_players =pd.concat([clusters, source], axis=1)
clustered_players = clustered_players.drop(columns=["Unnamed: 0"])

# get the number of players in each cluster
stats = clustered_players[["Cluster", "PTS"]].groupby(["Cluster"]).agg(["count"])

# ignore warnings for the polygone display
warnings.filterwarnings("ignore")

print("--- %s seconds ---" % round(time.time() - start_time, 2))


#now let's print the overlapping polygones for each cluster
for i in clustered_players.Cluster.unique():   
    players_to_draw = clustered_players[clustered_players["Cluster"] == i]["Player"].tolist()
    print(str(len(players_to_draw))+" players to draw.")
    properties = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]
    performance_polygon_vs_player(players_to_draw, properties)