# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:31:25 2017

@author: Arnab
"""

## Initialisation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
import copy
from pprint import pprint



##For plotting the clusters 

def plot_clusters(df,means,colmap):
    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in means.keys():
        plt.scatter(*means[i],s=200, color=colmap[i],marker='o')
    plt.xlim(-1, 5)
    plt.ylim(-2, 3)
    plt.show()


##Assign mean randomly 

def initial_mean_assignment(df,k):
    np.random.seed(200)
    initmeans = {
        i+1: [np.random.uniform(-2,4), np.random.uniform(-1, 2)]
        for i in range(k)
    }
    return initmeans

## Distance Function

def distance(x1,y1,x2,y2,n):
    if n==0:
        return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    elif n==1:
        return 1-(x1*x2+y1*y2)/(np.sqrt(x1**2+y1**2)*np.sqrt(x2**2+y2**2))


## Assignment Stage

def One_iteration_clustering(df, centroids, colmap,dist_m):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = distance(df['x'],df['y'],centroids[i][0],centroids[i][1],dist_m)
    mean_to_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, mean_to_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    
    return df
    
## Update the mean values   

def update_means(df,means):
    for i in means.keys():
        means[i][0] = np.mean(df[df['closest'] == i]['x'])
        means[i][1] = np.mean(df[df['closest'] == i]['y'])
    return means 
    
    
## For clusting    

def kmeans_clustering(df,k,colmap,dist_m):
    
    means=initial_mean_assignment(df,k)
    i=0;
    while 1:
        old_means=copy.deepcopy(means)
        df = One_iteration_clustering(df, means, colmap,dist_m)
        means = update_means(df,means)
        i=i+1;
        print "Iteration"+str(i)
        if old_means==means:
            break
    return means    
    
#To compare the two distance metrics 

def cluster_comparision(df,colormap):
    
    eeuc=[]
    ecos=[]
    k=[2,3,4,5,6]
    er_euc = pd.DataFrame({'2': [0],'3': [0],'4': [0],'5': [0],'6': [0]})
    er_cos = pd.DataFrame({'2': [0],'3': [0],'4': [0],'5': [0],'6': [0]})
    for i in range(2,7):
        mean_euc=kmeans_clustering(df,i,colmap,0)
        for j in range(len(df.index)):
		er_euc['{}'.format(i)] += df['distance_from_{}'.format(df['closest'][j])][j]
        mean_cos=kmeans_clustering(df,i,colmap,1)
        for j in range(len(df.index)):
		er_cos['{}'.format(i)] += df['distance_from_{}'.format(df['closest'][j])][j]
        eeuc.append(er_euc['{}'.format(i)])
        print er_euc['{}'.format(i)]
        ecos.append(er_cos['{}'.format(i)])
        print er_cos['{}'.format(i)]
    plt.plot(k,eeuc)
    plt.plot(k,ecos)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Error')
    plt.title('Error variation with number of clusters')
    return df
    
## To illustrate kmeans clustering     
        
def kmean_illus_(df,k,colmap,dist_m):
    means=kmeans_clustering(df,k,colmap,dist_m)
    plot_clusters(df,means,colmap)
    print "Successfully clustered with k="+str(k)     
        
    
if __name__ == '__main__':    
    a = sio.loadmat('data.mat')
    data=a['h']
    #pprint(data)
    df = pd.DataFrame(data,columns=list('xy'))
    colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y', 7: 'k'}
    
    '''
    Code segment to see the illustration of the kmeans clustering 
    dist_m values selects which distance metric we want to use 
    dist_m = 0 for Euclidean distance 
    dist_m = 1 for Cosine distane 
    '''
    dist_m = 1
    k=3
    #kmean_illus(df,k,colmap,dist_m)
    
    '''
    Code segment to see the plot of the error with number of clusters
    Knee point observed is k=3  
    '''
    
    df=cluster_comparision(df,colmap)
    
    


