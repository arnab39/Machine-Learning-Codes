import numpy as np
from pprint import pprint

attribute={1:'Credit History',2:'Wealth',3:'Assets'}

def entropy(x):
    res = 0
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res
	
def Entropy_gain(y, x):
    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])
    return res
	
def divide(a):
    return {b: (a==b).nonzero()[0] for b in np.unique(a)}

def split(x, y):
    # If no split is required, just return the original set
    if len(set(y)) == 1 or len(y) == 0:
        return y

    # We get attribute that gives the highest entropy gain
    gain = np.array([Entropy_gain(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return 
    #the original set
    if np.all(gain < 1e-6):
        return y

    # We split using the divide function on the selected attribute
    sets = divide(x[:, selected_attr])
    #print sets    
    
    #Output dictionary
    res = {}
       
    #To loop arout the partitioned dictionary and recursively send the data 
    #subset of the given features     
    for k, v in sets.items():
        
        #Label subset
        y_subset = y.take(v, axis=0)
        #print y_subset
        
        #Data subset
        x_subset = x.take(v, axis=0)
        #print x_subset

        res["%s = %d" % (attribute[selected_attr+1], k)] = \
        split(x_subset, y_subset)

    return res
    
    
x1= [0, 0, 1, 1, 2, 2]
x2= [0, 1, 1, 0, 0, 1]
x3= [1, 0, 0, 0, 1, 1]
y = np.array ([0, 0, 1, 0, 1, 1])    

X = np.array([x1, x2, x3]).T
pprint(split(X, y))
