# written by Qun Yan
# 11/11/2015

import cv2, os
import numpy as np
from PIL import Image


# load the face detection Cascade
cascadePath = "/Users/yanqun/Courses/machine_vision/project2/faceRecognization/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

imageNum = 54*5;
threshold = 10000;
recognizer = cv2.face.createEigenFaceRecognizer(imageNum, threshold)
images, labels = [], []

def get_images_labels(path):
    folders = [os.path.join(path, f) for f in os.listdir(path) if f not in ('.DS_Store', 'test')]
    for folder in folders:
       # get labels
       nbr = int(os.path.split(folder)[1].replace("yaleB", ""))
       # get images
       image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f not in ('.DS_Store')]
       for image_path in image_paths:
            img = cv2.imread(image_path,0)
            image = np.array(img, 'uint8')
            images.append(image)
            labels.append(nbr)
            cv2.imshow("test", img)
            cv2.waitKey(50)

    return images, labels

path = '/Users/yanqun/Courses/machine_vision/project2/faceRecognization/data/ale4'
images, labels = get_images_labels(path)
print np.shape(images)
print labels
cv2.destroyAllWindows()

# Perform the training
recognizer.train(images, np.array(labels))

# Do the test with images in test folder
test_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('test') ]
for test_path in test_paths:
    test_image_paths = [os.path.join(test_path, f) for f in os.listdir(test_path) if f not in ('.DS_Store') ]
    for test_image_path in test_image_paths:
        img = cv2.imread(test_image_path, 0)
        cv2.imshow("test", img)
        cv2.waitKey(50)
        predict_image = np.array(img, 'uint8')
        x = recognizer.predict(predict_image)
        print test_image_path, x