#!/usr/bin/python

import sys
import numpy as np
from sklearn.cluster import Birch

# n arguments (counting script name): len(sys.argv)
# list arguments: str(sys.argv) sys.argv[1] = file name

def gausssmoothing (rawcoveragematrix,shape=(3,3),sigma=1):
    def gaussfilt2D(shape=shape,sigma=sigma):
        m,n=[(edge-1)/2 for edge in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        array=np.exp(-(x*x+y*y)/(2*sigma*sigma))
        array[array<np.finfo(array.dtype).eps * array.max()] =0
        sumarray=array.sum()
        if sumarray !=0:
            array/=sumarray
        return array
    avmatrix=np.zeros_like(rawcoveragematrix)
    paddedrawcoveragematrix=np.concatenate((np.array([rawcoveragematrix[2,:] for i in range(int((shape[0]-1)/2))]),rawcoveragematrix,np.array([rawcoveragematrix[-1,:] for i in range(int((shape[0]-1)/2))])))
    paddedrawcoveragematrix=np.pad(paddedrawcoveragematrix,((0,0),(int((shape[0]-1)/2),int((shape[0]-1)/2))),'constant',constant_values=np.nan)
    for i in range(int((shape[0]-1)/2),int(len(rawcoveragematrix)+(shape[0]-1)/2)):
        print (i,'i')
        for j in range(int((shape[0]-1)/2),int(len(rawcoveragematrix[0])+(shape[0]-1)/2)):
            box=np.ma.masked_invalid(paddedrawcoveragematrix[int(i-(shape[0]-1)/2):int(i+(shape[0]-1)/2+1),int(j-(shape[0]-1)/2):int(j+(shape[0]-1)/2+1)])
            avmatrix[int(i-(shape[0]-1)/2),int(j-(shape[0]-1)/2)] = np.nansum(np.multiply(box,gaussfilt2D()))
    return (avmatrix)


def scalingto100range(input):
    a_scaled=np.zeros_like(input)
    for i in range(0,len(input)):
        for j in range (len(input[i])):
            a_scaled[i][j]=(input[i][j]/np.sum(input[:,j]))*100
    return (a_scaled)

def getbirch(input):
    brc = Birch(n_clusters=80)
    input_noNA=np.nan_to_num(input, nan=0, posinf=100, neginf=0)
    brc.fit(np.transpose(input_noNA))    
    means=np.zeros(shape=(16,80))
    for i in range(80):
        tmp=input_noNA[:,brc.labels_==i]
        tmp1=tmp.mean(axis=1)
        means[:,i]=tmp1

    np.savetxt(sys.argv[3]+".mean.birch.mat", np.transpose(means), fmt='%.8e', delimiter='\t')
    np.savetxt(sys.argv[3]+".mean.birch.label", brc.labels_, fmt='%d', delimiter='\t')


print("Processing", sys.argv[1])
print("\treading")
data_values=np.loadtxt(sys.argv[1], delimiter="\t")
data_values_np=np.array([*zip(*data_values)])
print("\tsmoothing")
data_values_smooth=gausssmoothing(data_values_np)
print("\tscaling")
data_values_smooth_scaled=scalingto100range(data_values_smooth)
print("\tsaving")
np.savetxt(sys.argv[2],[*zip(*data_values_smooth_scaled)],delimiter="\t")
print("\trunning Birch clustering")
getbirch(data_values_smooth_scaled)
