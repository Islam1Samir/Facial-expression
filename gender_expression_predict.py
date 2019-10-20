import cv2
import math
import numpy
import numpy as np
from matplotlib import pyplot as plt
import glob
import pickle


def major_vote(arr):
    cntArraay = np.zeros(8,np.int)
    for i in arr:
        cntArraay[i-1] +=1

    maxValue = np.argmax(cntArraay)
    return maxValue


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

filename = 'finalized_modelhof.sav'
svm = pickle.load(open(filename, 'rb'))

filename = 'finalized_modelgander.sav'
clfG = pickle.load(open(filename, 'rb'))
G = ['female','male']



target = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
targetG = ['female', 'male']
font = cv2.FONT_HERSHEY_SIMPLEX
arr = [0]*10
cnt = 0


VideoName = '01-01-08-01-01-02-01.mp4'


Video = cv2.VideoCapture(VideoName)
##Video = cv2.VideoCapture(0)  ##open live cam
ret, frame1 = Video.read()
wid, hig, zza = frame1.shape
if ret is True:


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
            hsv1[..., 0] = ang * 180 / np.pi / 2
            hsv1[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            faces = face_cascade.detectMultiScale(GrayScale, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (255, 255, 0), 2)
                img1 = hsv1[y:y + h, x:x + w]
                img1 = cv2.resize(img1, (64, 64))

                windowsize_r = 8
                windowsize_c = 8
                HOF = numpy.zeros((8, 8, 2, 10), dtype=int)

                # Crop out the window and calculate the histogram
                for r in range(0, img1.shape[0] - windowsize_r + 1, windowsize_r):
                    for c in range(0, img1.shape[1] - windowsize_c + 1, windowsize_c):
                        window = img1[r:r + windowsize_r, c:c + windowsize_c]
                        HOF[int(r / 8)][int(c / 8)][0] = np.histogram(window[..., 0], bins=np.arange(0, 256, 25.5))[0]
                        HOF[int(r / 8)][int(c / 8)][1] = np.histogram(window[..., 2], bins=np.arange(0, 256, 25.5))[0]


                hof = np.reshape(HOF, (1,1280))
                p = svm.predict(hof)
                arr[cnt%10]=p[0]
                v = major_vote(arr)

                cnt+=1
                cv2.putText(frame2, target[v], (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

                img1 = cv2.resize(img1, (64, 128))
                hog = cv2.HOGDescriptor()
                h = hog.compute(img1)
                h = np.reshape(h, (1, -1))

                p = clfG.predict(h)
                cv2.putText(frame2, G[int(p[0])], (x, y-20), font, 1, (200, 0, 0), 3, cv2.LINE_AA)


            cv2.imshow('j',frame2)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            prvs = next
        else:
            break;

Video.release()
cv2.destroyAllWindows()

