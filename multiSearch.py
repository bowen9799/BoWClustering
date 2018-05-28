#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from math import sqrt, ceil

from sklearn import preprocessing
import shutil

from pylab import *
from PIL import Image
from rootsift import RootSIFT
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from bic import compute_bic 

# calculate time elapsed
start_time = time.time()

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--pool", help="Path to query image pool", required="True")
args = vars(parser.parse_args())

# Get query image path
pool_path = args["pool"]

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

imlet_names = os.listdir(pool_path)
imlet_paths = []
for imlet_name in imlet_names:
    imlet_path = os.path.join(pool_path + imlet_name)
    imlet_paths += [imlet_path]

highClusters = [] 
lowClusters = []
for imlet_path in imlet_paths:
    clusters = searchSingle(imlet_path, image_paths, floor, ceiling)
    highClusters.append(clusters[0])
    lowClusters.append(clusters[1])

select(highClusters, lowClusters)

def searchSingle(img, image_paths, floor, ceiling):
    '''
    returns:
        highScoreCluster, list of paths of pics with high score, such as
        >= 0.5; 
        lowScoreCluster, ibid.;
    '''
    im = cv2.imread(pool_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # rootsift - not boosting performance
    #rs = RootSIFT()
    #des = rs.compute(kpts, des)

    des_list.append((pool_path, des))   
        
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]

    # gather features
    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors,voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features*idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    print "score: ", score
    rank_ID = np.argsort(-score)
    print "rank matrix: ", rank_ID[0]

    clusters = []

    for i, ID in enumerate(rank_ID[0][0:len(image_paths)]):
        print "ID = ", ID, " Score = ", score[0][ID], "\r"

        # manually cluster by score
        # Returns list of lists where each sublist is a cluster of IDs, and length of the master list is numClusters.
        if score[0][ID] >= ceiling:
            print "Putting picture %d into HIGH" % ID
            clusters[0].append(ID)
        elif score[0][ID] <= floor:
            print "Putting picture %d into LOW" % ID
            clusters[1].append(ID)

    return clusters

def select(highScoreClusters, lowScoreClusters):
    '''
    returns list of IDs of pics unioned as high-score clusters without intersection of the low-score clusters
    ''' 
    unionHigh = []
    unionLow = []
    for cluster in highScoreClusters:
        unionHigh += cluster
    for cluster in lowScoreClusters:
        unionLow += cluster
    return [i for i in unionHigh and i not in unionLow]

def archive(selected):
    return



# initiate clusters for picture ID
clusters = [[] for i in range(5)]

for i, ID in enumerate(rank_ID[0][0:409]):
    print "ID = ", ID, " Score = ", score[0][ID], "\r"

    # manually cluster by score
    # Returns list of lists where each sublist is a cluster of IDs, and length of the master list is numClusters.
    if score[0][ID] > 0.5:
        print "Putting picture %d into ScoreRange 0" % ID
        clusters[0].append(ID)
    elif score[0][ID] > 0.33:
        print "Putting picture %d into ScoreRange 1" % ID
        clusters[1].append(ID)
    elif score[0][ID] > 0.25:
        print "Putting picture %d into ScoreRange 2" % ID
        clusters[2].append(ID)
    elif score[0][ID] > 0.20:
        print "Putting picture %d into ScoreRange 3" % ID
        clusters[3].append(ID)
    elif score[0][ID] <= 0.20 and score[0][ID] >= 0:
        print "Putting picture %d into ScoreRange 4" % ID
        clusters[4].append(ID)
    else: 
        print "INVALID SCORE"

    # if score[0][ID] <= 0.3:
    #     print "Score <= 30%, terminating..."
    #     break
    # clusters of picture IDs marked to be similar to the query pic
    # list1.append(ID)

# before showing the results, print the time elapsed for the program
elapsed_time = time.time() - start_time
print "Time elapsed = ", elapsed_time

# write stats to report file in the root folder
f = open("report.txt", "w+")
for ID in list1:
    f.write("The score of Picture ")
    f.write(image_paths[ID])
    f.write(" = ")
    f.write(str(score[0][ID]))
    f.write("\n")
f.write("Time elapsed = ")
f.write(str(elapsed_time))
f.write("s\n")
f.close()

def clusterFiles(clusters, folderName):
    '''
    Put pictures of each cluster to respective cluster files.
    '''
    print "Putting these into files: ", clusters
    path, file = os.path.split(pool_path)
    if type(clusters) == 'numpy.ndarray':
        r = range(len(clusters.tolist()))
    else:
        r = range(len(clusters))

    for idx in r:
        newPath = path + folderName + str(idx)
        if os.path.exists(newPath):
            print folderName, " file exists; deleting them anyways"
            shutil.rmtree(newPath)
        print folderName, " path = " + newPath
        os.makedirs(newPath)
        if type(clusters[idx]) == 'numpy.ndarray':
            l = clusters[idx].tolist()
        else:
            l = clusters[idx]
        for picID in l:
            # shutil.move(image_paths[picID], newPath)
            shutil.copy(image_paths[picID], newPath)
    return

def clusterFilesK(ndarr, folderName, bestK):
    path, file = os.path.split(pool_path)
    l = ndarr.tolist()
    paths = []
    for idx in range(bestK):
        newPath = path + folderName + str(idx)
        if os.path.exists(newPath):
            print folderName, " file exists; deleting them anyways"
            shutil.rmtree(newPath)
        os.makedirs(newPath)
        paths.append(newPath)
    for picID in range(len(l)):
        clusterID = l[picID]
        print "X-Means: moving picture %d into cluster %d" % (picID, clusterID)
        shutil.copy(image_paths[picID], paths[clusterID])

# Cluster pictures only after everything else is done.
clusterFiles(clusters, '\ScoreRange')
clusterFilesK(KMeans[bestK - 1].labels_, '\Cluster', bestK)

# create true folder
# newpath = r'C:\Users\bowen.liu\Desktop\image-retrieval\bag-of-words-python-dev-version\dataset\cluster294\true'
# if not os.path.exists(newpath):
#     os.makedirs(newpath)

# move similar pics to the true folder
# for ID in list1:
#     shutil.move(image_paths[ID], newpath)