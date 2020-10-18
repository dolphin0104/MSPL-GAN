import time

start = time.time()

import argparse
import cv2
import itertools
import os

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
networkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
imgDim = 96

start = time.time()
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, imgDim)

img_dir = ''

detect_results = 0

for img_file in os.listdir(img_dir):
    imgPath = os.path.join(img_dir,img_file)
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    # face detection
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is not None:
        detect_results += 1
print(img_dir)
print(float(detect_results)/len(os.listdir(img_dir)))
