# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:58:11 2017

@author: hp pc
"""

import scipy.io as sio
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

a = sio.loadmat('data.mat')
b = a['h']

df = pd.DataFrame(b,columns=list('xy'))
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y', 6: 'k'}

euc_error = pd.DataFrame({'2': [0],'3': [0],'4': [0],'5': [0],'6': [0]})
cos_error = pd.DataFrame({'2': [0],'3': [0],'4': [0],'5': [0],'6': [0]})

def plotting(df, centroids, colmap):
#    fig = plt.figure(figsize=(5, 5))
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.2, edgecolor='k')
    for i in centroids.keys():
        plt.scatter(*centroids[i], color=colmap[i])
    plt.xlim(-1, 5)
    plt.ylim(-2, 3)
    plt.show()


def dist_calc(x1, x2, y1, y2, n):
    dist = 0
    if n == 0:
        dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
    elif n ==1:
        dist = 1 - ((x1*x2 + y1*y2)/(np.sqrt((x1**2 + y1**2)*(x2**2 + y2**2))))
    return dist

def init_centroid(k):
    np.random.seed(200)
    centroids = {
            i+1: [random.uniform(-0.3, 4.0), random.uniform(-1.4, 2.0)]
            for i in range(k)
            }
    print(centroids)
    return centroids
    
def first_assignment(df, centroids, colmap, dist_type):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = dist_calc(df['x'], centroids[i][0], df['y'], centroids[i][1], dist_type)
            
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update_centroid(df, centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids

def clustering(df, k, colmap, dist_type):
    centroids = init_centroid(k)
    i = 0
    while(1):
        prev_set = centroids
        print(prev_set)
        df = first_assignment(df, centroids, colmap, dist_type)
        centroids = update_centroid(df, centroids)
        print(centroids)
        i=i+1;
        print("Iteration"+str(i))
        if prev_set==centroids:
            break
    return centroids

def error_calc(df):
    e_euc =[]
    e_cos =[]
    k = [2,3,4,5,6]
    
    for i in range(2,7):
        for j in range(len(df.index)):
            euc_error['{}'.format(i)] += df['distance_from_{}'.format(df['closest'][j])][j]
        for j in range(len(df.index)):
            cos_error['{}'.format(i)] += df['distance_from_{}'.format(df['closest'][j])][j]
        e_euc.append(euc_error['{}'.format(i)])
        print(euc_error['{}'.format(i)])
        e_cos.append(cos_error['{}'.format(i)])
        print(cos_error['{}'.format(i)])
    plt.plot(k,e_euc)
    plt.plot(k,e_cos)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Error')
    plt.title('Error variation with number of clusters')
    return df

def k_means(df, k, colmap, dist_type):
    centroids = clustering(df, k, colmap, dist_type)
    plotting(df, centroids, colmap)


k = 4
dist_type = 1

k_means(df, k, colmap, dist_type)

#df = error_calc(df)

    