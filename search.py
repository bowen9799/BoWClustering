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
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []

im = cv2.imread(image_path)
kpts = fea_det.detect(im)
kpts, des = des_ext.compute(im, kpts)

# rootsift - not boosting performance
#rs = RootSIFT()
#des = rs.compute(kpts, des)

des_list.append((image_path, des))   
    
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


# ============ perform BIC ============ #
# def compute_bic(kmeans,X):
#     """
#     Computes the BIC metric for a given clusters

#     Parameters:
#     -----------------------------------------
#     kmeans:  List of clustering object from scikit learn

#     X     :  multidimension np array of data points

#     Returns:
#     -----------------------------------------
#     BIC value
#     """
#     # assign centers and labels
#     centers = [kmeans.cluster_centers_]
#     labels  = kmeans.labels_
#     #number of clusters
#     m = kmeans.n_clusters
#     # size of the clusters
#     n = np.bincount(labels)
#     #size of data set
#     N, d = X.shape

#     #compute variance for all clusters beforehand
#     print type(X)
#     cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
#              'euclidean')**2) for i in range(m)])

#     const_term = 0.5 * m * np.log(N) * (d+1)

#     BIC = np.sum([n[i] * np.log(n[i]) -
#                n[i] * np.log(N) -
#              ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
#              ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

#     return(BIC)



# IRIS DATA
# iris = sklearn.datasets.load_iris()
# X = iris.data[:, :4]  # extract only the features
Xlist = []
for x in score[0]:
    Xlist.append([x])
X = np.array(Xlist)
for n in X:
    n[0] *= 1000000 # increase difference
print "Scores for X-Means: ", X
# Xs = StandardScaler().fit_transform(X)
# Y = iris.target

ks = range(1,21)

# run ks times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]

# now run for each cluster the BIC computation
max = -sys.maxint - 1
maxIdx = 0
# BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]
for i in range(len(KMeans)):
    curr = compute_bic(KMeans[i], X)
    print "BIC = ", curr, " when using %d clusters\r" % (i + 1)
    if curr > max:
        max = curr
        maxIdx = i

bestK = maxIdx + 1
print "Best K = ", bestK, "with BIC = %d\r" % max
print "Best K labels", KMeans[bestK - 1].labels_


list1 = []

# Visualize the results
figure()
gray()
subplot(5,4,1)
imshow(im[:,:,::-1])
axis('off')

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


    print "plotting pic with path %s", image_paths[ID]
    img = Image.open(image_paths[ID])
    gray()
    # subplot(5,4,i+5)
    subplot(21, 21, i + 21)
    # subplot(ceil(sqrt(len(list1))) + 1, ceil(sqrt(len(list1))) + 1, i + ceil(sqrt(len(list1))) + 1)
    imshow(img)
    axis('off')

# before showing the results, print the time elapsed for the program
elapsed_time = time.time() - start_time
print "Time elapsed = ", elapsed_time

show()

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
    path, file = os.path.split(image_path)
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
    path, file = os.path.split(image_path)
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