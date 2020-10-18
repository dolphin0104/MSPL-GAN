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


def getRep(imgPath):    
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    
    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    
    if bb is not None:
        start = time.time()
        alignedFace = align.align(imgDim, rgbImg, bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)        
        if alignedFace is not None:    
            start = time.time()
            rep = net.forward(alignedFace)    
            return rep


gt_gallery = ''
img_dir = ''

gt_files = []
for root, dirs, files in os.walk(gt_gallery):
    for f in files:
        gt_files.append(os.path.join(root, f))

img_files = []
for root, dirs, files in os.walk(img_dir):
    for f in files:
        img_files.append(os.path.join(root, f))

top1_acc = 0
top3_acc = 0
top5_acc = 0
total_num = 0

for gt_imgs in gt_files:
    gt_cls = int(os.path.splitext(os.path.split(gt_imgs)[0])[-2][-1])
    dist[str(gt_cls)] = []
    feat1 = getRep(gt_imgs)
    
    dist = dict()
    for tar_imgs = in img_files:
        tar_cls = int(os.path.splitext(os.path.split(tar_imgs)[0])[-2][-1])
        feat2 = getRep(tar_imgs)
        d = feat1 - feat2
        distance = np.dot(d, d)
        dist[tar_cls]=distance                 
        total_num += 1
        
    sorted_dist = sorted(dist.items(), key=lambda x: x[1])[:5]     
    for k in range(5):
        if int(sorted_dist[k][0]))==gt_cls:
            if k ==0:
                top1_acc += 1                            
            if k <=2:
                top3_acc += 1                
            if k <=4:
                top5_acc += 1
        
print(float(top1_acc)*100/total_num)
print(float(top3_acc)*100/total_num)
print(float(top5_acc)*100/total_num)