from sklearn import svm

import pickle
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import glob

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

traningDatasetHof2 = []
labelHof2 = []

path = "/content/Actor_01/*"
Videos = glob.glob(path)
path = "/content/Actor_02/*"
Tmp = glob.glob(path)
Videos.extend(Tmp)
path = "/content/Actor_03/*"
Tmp = glob.glob(path)
Videos.extend(Tmp)
path = "/content/Actor_04/*"
Tmp = glob.glob(path)
Videos.extend(Tmp)
path = "/content/Actor_05/*"
Tmp = glob.glob(path)
Videos.extend(Tmp)
path = "/content/Actor_06/*"
Tmp = glob.glob(path)
Videos.extend(Tmp)

print(len(Videos))
for VideoName in Videos:
    print(VideoName)
    Video = cv2.VideoCapture(VideoName)
    ret, frame1 = Video.read()
    if ret is False:
        continue;
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv1 = np.zeros_like(frame1)
    hsv1[..., 1] = 255

    while Video.isOpened():
        flag, frame2 = Video.read()
        if flag is True:

            img = frame2
            GrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            next = GrayScale
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv1[..., 0] = cv2.normalize(ang * 180 / np.pi / 2, None, 0, 255, cv2.NORM_MINMAX)
            hsv1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            faces = face_cascade.detectMultiScale(GrayScale, 1.3, 5)
            for (x, y, w, h) in faces:

                img1 = hsv1[y:y + h, x:x + w]
                img1 = cv2.resize(img1, (64, 64))

                windowsize_r = 8
                windowsize_c = 8
                HOF = np.zeros((8, 8, 2, 10), dtype=int)

                # Crop out the window and calculate the histogram
                for r in range(0, img1.shape[0] - windowsize_r + 1, windowsize_r):
                    for c in range(0, img1.shape[1] - windowsize_c + 1, windowsize_c):
                        window = img1[r:r + windowsize_r, c:c + windowsize_c]
                        HOF[int(r / 8)][int(c / 8)][0] = np.histogram(window[..., 0], bins=np.arange(0, 256, 25.5))[0]
                        HOF[int(r / 8)][int(c / 8)][1] = np.histogram(window[..., 2], bins=np.arange(0, 256, 25.5))[0]

                hof = np.reshape(HOF, 1280)
                traningDatasetHof2.append(hof)
                labelHof2.append(int(VideoName[24:26]))

            prvs = next
        else:
            break;

    Video.release()
    cv2.destroyAllWindows()

x, y = np.shape(traningDatasetHof2)
trainDatasetHof2 = np.reshape(traningDatasetHof2, (x, y))
trainDatasetHof2 = np.float32(trainDatasetHof2)
labelHof2 = np.array(labelHof2)

np.save('traindatahof', trainDatasetHof2)
np.save('labelhof', labelHof2)

clf = svm.SVC(gamma='scale')
clf.fit(trainDatasetHof2, labelHof2)
filename = 'finalized_modelhof.sav'
pickle.dump(clf, open(filename, 'wb'))
print('saved')
