import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

data = scipy.io.loadmat('/content/wiki/wiki.mat')

labels = data['wiki'][0, 0][3]
bbox = data['wiki'][0, 0][5]

paths = data['wiki'][0, 0][2]

nd = np.where((data['wiki'][0, 0][1] > 2008) & ~(np.isnan(labels)) & (data['wiki'][0, 0][6] > 4) & (
    np.isnan(data['wiki'][0, 0][7])))
paths = paths[nd]
bbox = bbox[nd]
labels = labels[nd]

paths = [c[0] for c in paths]
bbox = [c[0] for c in bbox]
bbox = np.array(bbox)

print((bbox).shape)
train_data = []
for i in range(len(paths)):
    p = paths[i]
    ##print(bbox[i][0,0])
    pp = '/content/wiki/' + paths[i]

    im = np.array(Image.open(pp).convert('L'))

    ##cv2.rectangle(im, (int(bbox[i,0]),int(bbox[i,1])), (int(bbox[i,2]),int(bbox[i,3])), (255,0,0), 2)
    im = im[int(bbox[i, 1]):int(bbox[i, 3]), int(bbox[i, 0]):int(bbox[i, 2])]
    im = cv2.resize(im, (64, 128))
    hog = cv2.HOGDescriptor()
    h = hog.compute(im)
    train_data.append(np.array(h))

X_train, X_test, y_train, y_test = train_test_split(train_data,labels,test_size = 0.2,random_state = 0)
print(len(X_train),len(X_test),len(y_train),len(y_test))
print(len(y_train[y_train==0]),len(y_train[y_train==1]))
X_train = np.squeeze(np.array(X_train))
X_test = np.squeeze(np.array(X_test))
##print(np.squeeze(X_train).shape)
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)

predic =  clf.predict(X_test)

score = accuracy_score(predic,y_test)
print(score)
filename = 'finalized_modelgander.sav'
pickle.dump(clf, open(filename, 'wb'))
