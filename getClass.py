# getClass.py
# Written by Qun Yan
# 11/07/2015

import argparse as ap
import cv2

import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

sift = cv2.xfeatures2d.SIFT_create()
des_list = []

# the number of test images
testNum = 50;

# find keypoints and the descriptor in test image
for i in xrange(testNum):
    img = cv2.imread('/Users/yanqun/Courses/machine_vision/project2/bag-of-words-master/data/test3/{}.jpg'.format(i+1))
    if img == None:
        print "No such file {}\nCheck if the file exists"
        exit()
    print i

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, des) = sift.detectAndCompute(gray, None)
    des_list.append(des)



# Stack all the descriptors vertically in a numpy array
descriptors = des_list[1]
for descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

im_features = np.zeros((testNum, k), "float32")
for i in xrange(testNum):
    words, distance = vq(des_list[i],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*testNum+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
im_features = stdSlr.transform(im_features)

# Perform the predictions
predictions =  [classes_names[i] for i in clf.predict(im_features)]

# visualize the result
for i in xrange(testNum):
    image = cv2.imread('/Users/yanqun/Courses/machine_vision/project2/bag-of-words-master/data/test3/{}.jpg'.format(i+1))
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    pt = (0, 3 * image.shape[0] // 4)
    cv2.putText(image, predictions[i], pt ,cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
    cv2.imshow("Image", image)
    cv2.waitKey(30000)


