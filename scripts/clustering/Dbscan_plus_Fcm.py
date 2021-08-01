#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import sys
#import seaborn as sns
import sklearn
import warnings
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

sys.path.append("/home/elie/Documents/MoneyballReloaded/scripts")
from Polygone import performance_polygon_vs_player
#sns.set()

# need to install kneebow and mlxtend

df = pd.read_csv('/home/elie/Documents/MoneyballReloaded/csv/players_stats.csv')
player_names = df["Player"]

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])

results = pd.DataFrame(data = None, columns = ['epsilon' , 'min_size', 'score'],dtype=np.float64) 


for var_portion in np.arange(start = 0.6,stop=0.95,step=0.05,dtype=np.float64):
    pca = PCA(n_components=var_portion, svd_solver = 'full')
    pcabis = pca.fit(clustering_df)
    reducedDataSet = pcabis.transform(clustering_df)
    for eps in np.arange(start = 0.05,stop=0.95,step=0.01,dtype=np.float64):
        for size in np.arange(start = 2,stop=10,step=1,dtype=np.float64): 
            m = DBSCAN(eps=eps, min_samples=size)
            m.fit(reducedDataSet)
            if(max(m.labels_)>1):
                score = sklearn.metrics.silhouette_score(clustering_df,m.labels_)
                results = results.append({'var_portion' : var_portion, 'epsilon' : eps , 'min_size' : size , 'score' : score, 'nb_clusters' : max(m.labels_)+1}, ignore_index=True)

results = results.sort_values(by=["score"], ascending = False)
results.to_csv("/home/elie/Documents/MoneyballReloaded/csv/silhouette_search.csv", sep =';')

optimal_parameters = results.head(1)
optimal_parameters

pca_value = optimal_parameters.iloc[0]["var_portion"]
epsilon = optimal_parameters.iloc[0]["epsilon"]
min_size = optimal_parameters.iloc[0]["min_size"]

pca = PCA(n_components=pca_value, svd_solver = 'full')
pcabis = pca.fit(clustering_df)
dataSet = pcabis.transform(clustering_df)
model = DBSCAN(eps=epsilon, min_samples=min_size)
model.fit(dataSet)
result = pcabis.inverse_transform(dataSet)
res = np.zeros((0,3))
dbscan_cluster = pd.DataFrame(res)
number_of_players = df.shape[0]
for k in range(number_of_players):
    row = [[df['Player'].values[k], model.labels_[k], df["Pos"].values[k]]]
    dbscan_cluster = dbscan_cluster.append(row)
dbscan_cluster.columns = ["Player", "Cluster", "Pos"]



nb_players_per_cluster_dbscan = 4

pca = PCA(n_components=0.85, svd_solver = 'full')
pcabis = pca.fit(clustering_df)
dataSet = pcabis.transform(clustering_df)
model = DBSCAN(eps=0.22, min_samples=nb_players_per_cluster_dbscan)
model.fit(dataSet)
result = pcabis.inverse_transform(dataSet)
res = np.zeros((0,3))
dbscan_cluster = pd.DataFrame(res)
for k in range(df.shape[0]):
    row = [[df['Player'].values[k], model.labels_[k], df["Pos"].values[k]]]
    dbscan_cluster = dbscan_cluster.append(row)
    #cluster = cluster.sort_values(by=[1], ascending = False)
dbscan_cluster.columns = ["Player", "Cluster", "Pos"]




dbscan_cluster = dbscan_cluster.drop(columns="Pos")

# we remove the noise from the dbscan clusters
dbscan_cluster = dbscan_cluster[dbscan_cluster["Cluster"] != -1]

# we also considere that the maximum number of player of a consistent cluster is about 4
# we need to consider as noise the players in big cluster
nb_of_players_per_cluster = dbscan_cluster.groupby("Cluster").agg("count")
too_big_clusters = nb_of_players_per_cluster[nb_of_players_per_cluster["Player"] > nb_players_per_cluster_dbscan]["Player"]
dbscan_cluster = dbscan_cluster.loc[~dbscan_cluster['Cluster'].isin(too_big_clusters.index)]

nb_of_clusters_from_filtered_dbscan = len(dbscan_cluster.Cluster.unique())

# lets see who is clustered and who's not
clustered_players = dbscan_cluster["Player"]
unclustered_players = df.loc[~df["Player"].isin(clustered_players)]["Player"]

# highest cluster # found in dbscan
nb_clusters_from_dbscan = max(dbscan_cluster["Cluster"])

# we keep the interesting value
df_fcm = df[['Player', 'TRB', 'PTS', 'AST', 'DWS', 'TS%', "3PA", "OWS","USG%"]]

# we only keep the unclustered player
df_fcm = pd.merge(df_fcm, unclustered_players, on="Player")

# we keep the players name for later
players_name = df_fcm["Player"]
# we remove the player column for the computation
df_fcm = df_fcm.loc[:,(df_fcm.columns != "Player")]

# Computation
#nb_cluster_fuzzy = round(len(unclustered_players.index)/nb_max_players_per_cluster)
nb_cluster_fuzzy = 35
fuzzy_kmeans = FuzzyKMeans(k=nb_cluster_fuzzy, m=1.1)

# we can also directly compute fcm on the data (and not on the dbscan noise)

fuzzy_kmeans.fit(df_fcm)
fuzzy_clusters = pd.DataFrame(fuzzy_kmeans.fuzzy_labels_)

# we add the players name back
fuzzy_clusters = pd.concat([players_name, fuzzy_clusters], axis=1)

print("At first we clustered "+str(len(dbscan_cluster.index))+" players with DSBCAN.")

final_clusters = dbscan_cluster

nb_max_players_per_cluster_fcm = 4


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
    sets["Cluster"] = nb_clusters_from_dbscan+i+1
    #add those lines to the previous results
    final_clusters = pd.concat([final_clusters, sets], axis=0)


# ignore warnings for the polygone display
warnings.filterwarnings("ignore")

nb_of_cluster_printed = 0
nb_of_players_clustered = 0

#now let's print the overlapping polygones for each cluster
for i in final_clusters.Cluster.unique():   
    players_to_draw = dbscan_cluster[dbscan_cluster["Cluster"] == i]["Player"].tolist()
    properties = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]
    if (len(players_to_draw) < 10):
        nb_of_cluster_printed+=1
        nb_of_players_clustered += len(players_to_draw)
        performance_polygon_vs_player(players_to_draw, properties)


print("We clustered "+str(nb_of_players_clustered)+" players with DBSCAN in "+str(nb_of_cluster_printed)+" clusters out of "+str(len(df.index))+" players.")
#print("Now we have "+str(len(final_clusters.index))+" players clustered out of "+str(len(clustering_df.index))+" players.")
