import cv2
import numpy as np

datapath = "/Users/yanqun/Courses/machine_vision/project2/bag-of-words-master/data"

# a function to call the path of images
def path(cls, i) :
    return "%s/%s/%d.jpg" % (datapath, cls, i + 1)

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm = 1, trees = 5)      # flann enums are missing, FLANN_INDEX_KDTREE=1
matcher = cv2.FlannBasedMatcher(flann_params, {})

bow_train = cv2.BOWKMeansTrainer(400)
bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)
#help(bow_train)
#help(bow_extract)

trainNum = 100;


for i in range(trainNum):
    img = cv2.imread(path("train2", i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, des) = extract.detectAndCompute(gray, None)
    bow_train.add(des)


# cluster descriptors to vocabulary
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)
print "bow voc", np.shape(voc)


traindata, trainlabels = [], []

class_id = 0
for x in xrange(5):
    trainlabels +=[class_id] * 20
    class_id += 1

for i in range(trainNum):
    img = cv2.imread(path("train2", i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, des) = extract.detectAndCompute(gray, None)
    traindata.extend(bow_extract.compute(gray, kps,des))

print "svm items", len(traindata), traindata[0]

svm = cv2.ml.SVM_create()
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

for i in range(50):
    img = cv2.imread(path("train2", i))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (kps, des) = extract.detectAndCompute(gray, None)
    f = bow_extract.compute(gray, kps,des)
    p = svm.predict(f)
    print i, ".jpg",  "\t", p[1][0][0]

