#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
import shutil

from PIL import Image
from rootsift import RootSIFT
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from bic import compute_bic 
import sys, csv, time
from random import randint
from histsimilar import compare_all

# calculate time elapsed
start_time = time.time()

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--pool", help="Path to query image pool", required=True)
parser.add_argument("-b", "--low_threshold", help="Floor value", required=False)
parser.add_argument("-t", "--high_threshold", help="Ceiling value", required=False)
parser.add_argument("-p", "--percentage", help="limit numbers", required=False)
parser.add_argument("-c", "--color_lowcut", help="colorwise lowcut", required=False)
parser.add_argument("-v", "--verbose")

args = vars(parser.parse_args())

# Get query image path
pool_path = args["pool"]
color_lowcut = float(args["color_lowcut"]) if args["color_lowcut"] else 0.2
low_threshold = float(args["low_threshold"]) if args["low_threshold"] else 0.15
high_threshold = float(args["high_threshold"]) if args["high_threshold"] else 0.55
verbose = True if args["verbose"] else False

if verbose:
    print "low_t = %d, high_t = %d, pool_path = %s" % (low_threshold, high_threshold, pool_path)

# Load the classifier, class names, scaler, number of clusters and vocabulary 
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SURF")
des_ext = cv2.DescriptorExtractor_create("SURF")

# score map to put into csv in the future
score_map = {}


def search_single(img):
    """
    returns:
        highScoreCluster, list of paths of pics with high score, such as
        >= 0.5; 
        lowScoreCluster, ibid.;
    """
    print "Searching similar pics for pool image %s...\n" % img
    im = cv2.imread(img)
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    # List where all the descriptors are stored
    des_list = list()

    # rootsift - not boosting performance
    # rs = RootSIFT()
    # des = rs.compute(kpts, des)

    des_list.append((img, des))   
        
    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]

    # gather features
    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features*idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    rank_id = np.argsort(-score)
    if verbose:
        print "==================search_single()==================\n"
        # print "score: ", score
        # print "rank matrix: ", rank_ID[0]

    clusters = [[],[],[]]

    for i, ID in enumerate(rank_id[0][0:len(image_paths)]):
        if verbose:
            print "ID = ", ID, " Name = ", image_paths[ID], " Score = ", score[0][ID], "\r"

        if args["percentage"]:
            count_mid = 0
            for i, ID in enumerate(rank_id[0][0:len(image_paths)]):

                # put score into score map
                if score_map.has_key(ID):
                    score_map[ID].append(score[0][ID])
                else:
                    # print "adding %d into scoremap for the first time" % ID
                    score_map[ID] = [score[0][ID]]

                if i <= len(image_paths) * 0.05:
                    # if score[0][ID] <= 0.42:
                    #         print "IMAGE " + img + " MAY NOT BE A GOOD CANDIDATE.\n"
                    #         sys.exit()
                    if verbose:
                        print "**HIGH** ID = ", ID, " Name = ", image_paths[ID], " Score = ", score[0][ID], "\r"
                    clusters[0].append(ID)
                elif score[0][ID] <= 0.15:
                    if verbose:
                        print "**LOW** ID = ", ID, " Name = ", image_paths[ID], " Score = ", score[0][ID], "\r"
                    clusters[1].append(ID)
                else:
                    count_mid += 1
                    clusters[2].append(ID)
            if verbose:
                print "MID CLUSTER SIZE = ", count_mid
            break

        # manually cluster by score
        # Returns list of lists where each sublist is a cluster of IDs, and length of the master list is numClusters.
        if score[0][ID] >= high_threshold:
            if verbose:
                print "Putting picture %s into HIGH" % image_paths[ID]
            clusters[0].append(ID)
        elif score[0][ID] <= low_threshold:
            if verbose:
                print "Putting picture %s into LOW" % image_paths[ID]
            clusters[1].append(ID)
        elif high_threshold > score[0][ID] > low_threshold: 
            if verbose:
                print "Putting picture %s into MID" % image_paths[ID]
            clusters[2].append(ID)
    
    if verbose:
        print "search_single() for image" + img + "generates %d lowpics, %d midpics and %d highpics\n" % (len(clusters[1]), len(clusters[2]), len(clusters[0]))
        print "search_single() for image" + img + "gives current score matrix" + score_map
    return clusters


def intersection(nested_list):
    while len(nested_list) > 1:
        l_0 = nested_list[0]
        l_1 = nested_list[1]
        nested_list[1] = [id for id in l_0 if id in l_1]
        del nested_list[0]
    res = nested_list[0]
    return res


def union(nested_list):
    res = []
    for cluster in nested_list:
        res += [id for id in cluster if id not in res]
    return res


def select(highScoreClusters, lowScoreClusters, mid_clusters, neg_colorwise):
    """
    returns list of IDs of pics unioned as high-score clusters without intersection of the low-score clusters
    """
    res_high = union(highScoreClusters)
    res_low = union(lowScoreClusters)
    # res_mid = intersection(mid_clusters)
    color_low = union(neg_colorwise)
    print "\nColorwise negative images: ", color_low

    # res_high - res_low - res_mid
    # res = [i for i in res_high if i not in res_low if i not in res_mid]
    # res += res_mid
    res = [i for i in res_high if i not in res_low 
    if os.path.split(image_paths[i])[1] not in color_low]

    # alt method: iterative filtering
    # res = res_high.append(res_mid)

    if verbose:
        print "\n ==================select()=================="
        # print "high & low score clusters: \n", highScoreClusters, "\n", lowScoreClusters
        print "length of res_high and res_low: \n", len(res_high), len(res_low)

    return res, res_low, color_low


def archive(selected_pics, low_pics, color_low, dir_path):
    if verbose:
        print "\n ==================archive()=================="
        print "Generating %d lowpics and %d highpics..." % (len(low_pics), len(selected_pics))
    if args["percentage"]:
        res_folder_name = "N=" + str(len(selected_pics)) + ", percentage=True, low_color_num=" + str(len(color_low))
    else:
        res_folder_name = "N=" + str(len(selected_pics)) + ",f=" + str(low_threshold) + ",c=" + str(high_threshold) + "low_color_num =" + str(len(color_low))
    new_path = os.path.split(dir_path)[0] + "/" + res_folder_name
    if os.path.exists(new_path):
        print new_path, "\nfile exists; replacing..."
        shutil.rmtree(new_path)
    low_pics_path = new_path + "/" + "below " + str(low_threshold)
    color_low_path = new_path + "/" + "color score under " + str(color_lowcut)
    os.mkdir(new_path)
    os.mkdir(low_pics_path)
    os.mkdir(color_low_path)
    for ID in selected_pics:
        shutil.copy(image_paths[ID], new_path)
    for ID in low_pics:
        shutil.copy(image_paths[ID], low_pics_path)
    for img_name in color_low:
        shutil.copy(os.path.split(dir_path)[0] + "/" + img_name, color_low_path)
    return


highClusters = [] 
lowClusters = []
mid_clusters = []
neg_colorwise = []
csv_header = ["Image Name"]
for imlet_name in os.listdir(pool_path):
    csv_header.append(imlet_name)
    imlet_path = os.path.split(pool_path)[0] + "/" + imlet_name
    if verbose:
        print imlet_path
    clusters = search_single(imlet_path)
    highClusters.append(clusters[0])
    lowClusters.append(clusters[1])
    mid_clusters.append(clusters[2])
    neg_colorwise.append([t[0] for t in compare_all(imlet_path) \
    if t[1] < color_lowcut])

res_pics, res_low, color_low = select(highClusters, lowClusters, mid_clusters, neg_colorwise)

# while (len(res_pics) < 50):
#     print "true pics < 50:", len(res_pics)
#     idx = 0
#     for n in range(100 - len(res_pics)):
#         clusteridx = idx % len(mid_clusters)
#         cand = mid_clusters[clusteridx][randint(0, len(mid_clusters[clusteridx]) - 1)]
#         for cluster in lowClusters:
#             if cand in cluster:
#                 break
#             else:
#                 res_pics.append(cand)
#         idx += 1
#
# while (len(res_low) < 50):
#     print "false pics < 50:", len(res_low)
#     idx = 0
#     for n in range(100 - len(res_low)):
#         clusteridx = idx % len(mid_clusters)
#         cand = mid_clusters[clusteridx][randint(0, len(mid_clusters[clusteridx]) - 1)]
#         for cluster in highClusters:
#             if cand in cluster:
#                 break
#             else:
#                 res_low.append(cand)
#         idx += 1

archive(res_pics, res_low, color_low, pool_path)

# write stats to report file in the root folder
# f = open(pool_path + "report.txt", "w+")
# for ID in similar_pics:
#     f.write("The score of Picture ")
#     f.write(image_paths[ID])
#     f.write(" = ")
#     f.write(str(score[0][ID]))
#     f.write("\n")
# f.close()

# put scores into csv report file
print "Generate CSV report with score map", score_map
if os.path.exists('\report.csv'):
    print "\rReplacing existing report.csv file"
    shutil.rmtree('\report.csv')
csv_path = pool_path + '/' + 'report.csv'
with open(csv_path, 'wb') as f:
    a = csv.writer(f, delimiter=',')
    a.writerow(csv_header)
    for key, val in score_map.items():
        content = [os.path.split(image_paths[key])[1]]
        content.extend(val)
        content.append(str(key in res_pics))
        if os.path.split(image_paths[key])[1] in neg_colorwise:
            content.append("CUTOFF")
            print "\nCutting off ", os.path.split(image_paths[key])[1]
        a.writerow(content)


# before showing the results, print the time elapsed for the program
elapsed_time = time.time() - start_time
print "Time elapsed = ", elapsed_time
