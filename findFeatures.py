#!/usr/local/bin/python2.7
#python findFeatures.py -t dataset/train/

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing
from rootsift import RootSIFT
import math
import time

from PIL import Image, ImageDraw
from pylab import *
import glob
from scipy.cluster.vq import *


def find_features(image_paths):
    """
    @:param image_paths: list of paths of images in source data
    :return: modifies bof.pkl
    """

    numWords = 1000

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SURF")
    des_ext = cv2.DescriptorExtractor_create("SURF")

    # List where all the descriptors are stored
    des_list = []

    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print "Extract SURF of %s image, %d of %d images" % (image_paths[i], i, len(image_paths))
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        # rootsift
        # rs = RootSIFT()
        # des = rs.compute(kpts, des)
        des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
    # downsampling = 1
    # descriptors = des_list[0][1][::downsampling,:]
    # for image_path, descriptor in des_list[1:]:
    #    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

        # findFeatures.py ORIGINAL CODE
    # Perform k-means clustering
    print "Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0])
    voc, variance = kmeans(descriptors, numWords, 1)
    print "\nDone k-means with voc = ", voc, " variance = ", variance

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), numWords), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    print "\nStart TF-IDF vectorization..."
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    print "\nPerform L2 normalization"
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    print "\ndump features..."
    joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)

    return


if __name__ == "__main__":
    # calculate time elapsed
    start_time = time.time()

    # Get the path of the training set
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
    args = vars(parser.parse_args())

    # Get the training classes names and store them in a list
    train_path = args["trainingSet"]
    # train_path = "dataset/train/"

    training_names = os.listdir(train_path)

    numWords = 1000

    # Get all the path to the images and save them in a list
    # image_paths and the corresponding label in image_paths
    image_paths = []
    for training_name in training_names:
        image_path = os.path.join(train_path, training_name)
        if os.path.isdir(image_path):
            print "skipping non-image: ", image_path
        continue
        image_paths += [image_path]

    # Create feature extraction and keypoint detector objects
    fea_det = cv2.FeatureDetector_create("SURF")
    des_ext = cv2.DescriptorExtractor_create("SURF")

    # List where all the descriptors are stored
    des_list = []

    for i, image_path in enumerate(image_paths):
        im = cv2.imread(image_path)
        print "Extract SIFT of %s image, %d of %d images" % (training_names[i], i, len(image_paths))
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        # rootsift
        # rs = RootSIFT()
        # des = rs.compute(kpts, des)
        des_list.append((image_path, des))

        # Stack all the descriptors vertically in a numpy array
    # downsampling = 1
    # descriptors = des_list[0][1][::downsampling,:]
    # for image_path, descriptor in des_list[1:]:
    #    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

        # findFeatures.py ORIGINAL CODE
    # Perform k-means clustering
    print "Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0])
    voc, variance = kmeans(descriptors, numWords, 1)
    print "\nDone k-means with voc = ", voc, " variance = ", variance

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), numWords), "float32")
    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Perform Tf-Idf vectorization
    print "\nStart TF-IDF vectorization..."
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Perform L2 normalization
    print "\nPerform L2 normalization"
    im_features = im_features * idf
    im_features = preprocessing.normalize(im_features, norm='l2')

    print "\ndump features..."
    joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)

    '''
    print "computing kmeans...\r"
    print len(descriptors[0])
    k = 2 # number of clusters
    descriptors = whiten(descriptors)
    centroids,distortion = kmeans(descriptors, k)
    print "done kmeans"
    code, distance = vq(descriptors, centroids)
    # plot clusters
    for c in range(k):
        ind = np.where(code==c)[0]
        print ind
        figure()
        gray()
        for i in range(minimum(len(ind), 5)):
            im = Image.open(image_paths[ind[i]])
            subplot(5,4,i+1)
            imshow(array(im))
            axis('equal')
            axis('off')
    show()
    '''

    # before showing the results, print the time elapsed for the program
    elapsed_time = time.time() - start_time
    print elapsed_time