# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:16:33 2020

@author: Aksel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.path as path
import matplotlib.patches as patches

PlayerStats="MP"

NormalizeData = pd.read_csv("./../csv/players_stats.csv", delimiter =",");

# a function which computes a performance polygon for a specific player using three parameters


def performance_polygon(PlayerName):
    Player=10*NormalizeData[NormalizeData.Player.eq(PlayerName)]

    # Player = AdDisp[AdDisp.Year.eq(2020)]

    properties = ['Offensive Win share', 'Defensive win share', 'AST','TS%', "TRB%", "PTS", "3PA", ]
    values = np.random.uniform(5,9,len(properties))

    values = [Player['OWS'], Player['DWS'], Player['AST'], Player["TS%"], Player["TRB%"], Player["PTS"], Player["3PA"]]
    matplotlib.rc('axes', facecolor = 'white')

    fig = plt.figure(figsize=(10,8), facecolor='white')

    axes = plt.subplot(111, polar=True)

    t = np.arange(0,2*np.pi,2*np.pi/len(properties))
    plt.xticks(t, [])

    points = [(x,y) for x,y in zip(t,values)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO,] + \
            [path.Path.LINETO,]*(len(values) -1) + \
            [ path.Path.CLOSEPOLY ]
    _path = path.Path(points, codes)
    _patch = patches.PathPatch(_path, fill=True, color='blue', linewidth=0, alpha=.1)
    axes.add_patch(_patch)
    _patch = patches.PathPatch(_path, fill=False, linewidth = 2)
    axes.add_patch(_patch)

    plt.scatter(points[:,0],points[:,1], linewidth=2,
                s=50, color='white', edgecolor='black', zorder=10)
    """
    maxi = max([Player.iloc[0,19]+1, Player.iloc[0,20]+1, Player.iloc[0,21]+1])
    if maxi < 10:
        plt.ylim(0,10)
    else:
        plt.ylim(0,maxi)
        """
    plt.ylim(0,10)
    for i in range(len(properties)):
        angle_rad = i/float(len(properties))*2*np.pi
        angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 10.75, properties[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

    plt.title("Statistics of "+PlayerName)
    plt.show()
    
    
def performance_polygon_vs_player(PlayersName, criterias):
    #properties = ['Offensive Win share', 'Defensive win share', 'AST','TS%', "TRB", "PTS", "3PA" ]
    values = np.random.uniform(5,9,len(criterias))
    colors = ["blue", "red", "green", "orange", "brown", "deeppink","sienna",
              "gold", "olivedrab", "mediumspringgreen", "navy", "plum", "cadetblue", "darkmagenta"
              , "black"]
    fig = plt.figure(figsize=(10,8), facecolor='white')
    for i in range (0,len(PlayersName)):
        Player=10*NormalizeData[NormalizeData.Player.eq(PlayersName[i])]
        #Player2=10*NormalizeData[NormalizeData.Player.eq(PlayerName[1])]
        
        # Player = AdDisp[AdDisp.Year.eq(2020)]
    

        values1 = [Player[item] for item in criterias]
        #values1 = [Player['OWS'], Player['DWS'], Player['AST'], Player["TS%"], Player["TRB"], Player["PTS"], Player["3PA"]]
        #values2 = [Player2['OWS'], Player2['DWS'], Player2['AST'], Player2["TS%"], Player2["TRB%"], Player2["PTS"], Player2["3PA"]]
        matplotlib.rc('axes', facecolor = 'white')

        axes = plt.subplot(111, polar=True)
        
        t = np.arange(0,2*np.pi,2*np.pi/len(criterias))
        plt.xticks(t, [])
    
        points = [(x,y) for x,y in zip(t,values1)]
        points.append(points[0])
        points = np.array(points, dtype=object)
        codes = [path.Path.MOVETO,] + \
                [path.Path.LINETO,]*(len(values) -1) + \
                [ path.Path.CLOSEPOLY ]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=False, color=colors[i], linewidth=0, alpha=.2)
        axes.add_patch(_patch)
        _patch = patches.PathPatch(_path, fill=False, edgecolor=colors[i], linewidth = 2, label=PlayersName[i])
        axes.add_patch(_patch)
        plt.scatter(points[:,0],points[:,1], linewidth=2,
                s=50, color='white', edgecolor='black', zorder=10)
    #plt.scatter(points[:,0],points[:,1], linewidth=2,s=50, color='white', edgecolor='black', zorder=10)
    plt.legend(loc="lower right",borderaxespad=-6)
    """
    maxi = max([Player.iloc[0,19]+1, Player.iloc[0,20]+1, Player.iloc[0,21]+1])
    if maxi < 10:
        plt.ylim(0,10)
    else:
        plt.ylim(0,maxi)
        """
    plt.ylim(0,10)
    for i in range(len(criterias)):
        angle_rad = i/float(len(criterias))*2*np.pi
        angle_deg = i/float(len(criterias))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 10.75, criterias[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

    plt.title("Performance polygon", pad = 50)
    # plt.savefig("polygone")
    plt.show()
    return fig


# list_of_player = ["Cody Zeller", "Willie Cauley-Stein", "Nerlens Noel", "Taj Gibson", "Ian Mahinmi"]
# properties = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]
#performance_polygon_vs_player(list_of_player, properties)