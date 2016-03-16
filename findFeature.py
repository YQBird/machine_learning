# findFeature.py
# Written by Qun Yan
# 11/07/2015

import argparse as ap
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

# same class name in a list
class_name = ["aeroplane",  "car", "mug", "cat", "house" ]

sift = cv2.xfeatures2d.SIFT_create()
des_list = []

# read all traing image and save descriptor in des_list
for i in xrange (1,101):
    img = cv2.imread('/Users/yanqun/Courses/machine_vision/project2/bag-of-words-master/data/train3/{}.jpg'.format(i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, des) = sift.detectAndCompute(gray, None)
    des_list.append(des)

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0]
for descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
# return k*n array
k = 200
voc, variance = kmeans(descriptors, k, 1)

# Calculate the histogram of features


trainNum = 100
class_id = 0
image_classes = []
for x in xrange(5):
    image_classes +=[class_id] * 20
    class_id += 1

im_features = np.zeros((trainNum, k), "float32")
for i in xrange(trainNum):
    words, distance = vq(des_list[i],voc)
    for w in words:
        im_features[i][w] += 1


# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*trainNum+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM
joblib.dump((clf, class_name, stdSlr, k, voc), "bof.pkl", compress=3)
