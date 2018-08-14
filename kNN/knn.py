import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt

# Split the data into training and test data
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for row in dataset:
            del row[3]
        max_attr=np.zeros(25)   
        #print dataset
        #size=len(dataset)-1
        #print size
        for x in range(1,len(dataset)):
            #print size
            for y in range(26):
                if dataset[x][y+2] == 'NA':
                    dataset[x][y+2] = 0
                '''
                    del dataset[x]
                    size=size-1
                    break
                print x,y 
                '''
                dataset[x][y] = float(dataset[x][y+2])
                if y < 25 and max_attr[y] < dataset[x][y] :
                    max_attr[y] = dataset[x][y]
        for row in dataset:
            del row[26:30]         
               
        for x in range(1,len(dataset)):
            for y in range(25):
                dataset[x][y] = dataset[x][y]/max_attr[y]
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
                
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)
    
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
    
    
def findOut(neighbors,k):
    out_sum=0    
    for x in range(k):
        out_sum = out_sum + neighbors[x][-1]  
    return out_sum/k    
    
                
if __name__ == '__main__':
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadDataset('nba_2013.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))   
    mse_list=[]
    k_list=[]
    for i in range(10):
        k=2*i+1
        mse=0
        for x in range(len(testSet)):
            neighbors = getNeighbors(trainingSet, testSet[x], k)
            result = findOut(neighbors,k)            
            mse+= (result-testSet[x][-1])**2
        k_list.append(k)
        mse/=len(testSet) 
        mse_list.append(mse)
    plt.plot(k_list,mse_list)
    plt.xlabel('K values')
    plt.ylabel('Mean Square error')
    plt.title('Mean Square error vs k plot')
    