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
parser.add_argument("-p", "--pool", help="Path for pool to query image", required=True)
parser.add_argument("-l", "--lowcut", help="Low bound to cut off false data", required=False)
args = vars(parser.parse_args())

# Get query image path
pool_path = args["pool"]
lowcut = args["lowcut"] if args["lowcut"] else 0.2

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SURF")
des_ext = cv2.DescriptorExtractor_create("SURF")

low_union = []


def x_means(image_path):
    """
    perform retrieval and k-means on score matrix and pick best k in range (30 * n / (200 + n)) +- 5
    :return: biggest cluster
    """

    # List where all the descriptors are stored
    des_list = []

    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # rootsift - not boosting performance
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)

    des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    print descriptors

    # gather features
    test_features = np.zeros((1, numWords), "float32")
    words, _ = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    # print "score: ", score
    rank_ID = np.argsort(-score)
    # print "rank matrix: ", rank_ID[0]

    for ID in rank_ID[0]:
        if score[0][ID] <= lowcut:
            if ID not in low_union:
                low_union.append(ID)

    x_list = []
    for x in score[0]:
        x_list.append([x])
    X = np.array(x_list)
    for n in X:
        n[0] *= 1000000  # increase difference
    print "Scores for X-Means: ", X
    # Xs = StandardScaler().fit_transform(X)
    # Y = iris.target

    # compute k range
    # ks = range(1, 21)
    bot_k = 30 * len(image_paths) / (200 + len(image_paths)) - 5
    ks = range(bot_k, bot_k + 10)

    # run ks times kmeans and save each result in the KMeans object
    KMeans = [cluster.KMeans(n_clusters=i, init="k-means++").fit(X) for i in ks]

    # now run for each cluster the BIC computation
    bic_max = -sys.maxint - 1
    max_idx = 0
    # BIC = [compute_bic(kmeansi,X) for kmeansi in KMeans]
    for i in range(len(KMeans)):
        curr = compute_bic(KMeans[i], X)
        print "BIC = ", curr, " when using %d clusters\r" % (i + bot_k)
        if curr > bic_max:
            bic_max = curr
            max_idx = i

    best_k = max_idx + bot_k
    # print "Best K = ", best_k, "with BIC = %d\r" % bic_max
    best_k_labels = KMeans[max_idx].labels_
    # print "Best K labels", KMeans[max_idx].labels_

    freq = {}
    for cluster_no in best_k_labels:
        if cluster_no in freq:
            freq[cluster_no] = freq[cluster_no] + 1
        else:
            freq[cluster_no] = 1
    max_cluster = 0
    max_size = -1
    for k, v in freq.items():
        if v > max_size:
            max_cluster = k
            max_size = v
    res = []
    for idx in range(len(best_k_labels)):
        if best_k_labels[idx] == max_cluster:
           res.append(idx)
    assert max_size == len(res)
    print "\nSize of the largest cluster = ", max_size

    path = os.path.split(pool_path)[0] + "\\" + "cluster_" + os.path.split(image_path)[1]
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    print "Adding cluster images for pool image ", image_path, " to path ", path
    for ID in res:
        print "Adding ", image_paths[ID]
        shutil.copy(image_paths[ID], path)

    return res  # biggest_cluster list

# generate ~(n/(a=4)) candidates
biggest_cluster_union = []
for img in os.listdir(pool_path):
    img_path = os.path.split(pool_path)[0] + "\\" + img
    for cand_img in x_means(img_path):
        if cand_img not in biggest_cluster_union:
            biggest_cluster_union.append(cand_img)

print "\nLow Union: ", [image_paths[ID] for ID in low_union]
result = [cand for cand in biggest_cluster_union if cand not in low_union]

new_path = os.path.split(pool_path)[0] + "\\" + "n=" + str(len(result))
if os.path.exists(new_path):
    shutil.rmtree(new_path)
os.mkdir(new_path)
for ID in result:
    shutil.copy(image_paths[ID], new_path)

print "Getting result with size ", len(result), " using lowcut ", lowcut

elapsed_time = time.time() - start_time
print "Time elapsed = ", elapsed_time