
"""
Created on Tue Nov 26 15:22:03 2019

@author: sergi
"""

def split(cluster):
    cores=18
    parr =[[]]*cores
    part =len(cluster)/cores
    
    for i in range(cores):
        parr[i] = cluster[i*part : (i+1)*part]
    if not(len(cluster) % 18 == 0):
        parr[17] += cluster[len(cluster)-(len(cluster)%18) : len(cluster)]
        
    return parr
