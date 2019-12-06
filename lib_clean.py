#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import math
import cv2 as cv
from matplotlib import pyplot as plt
from lib_levels import ImageLevels

ratio = 3
kernel_size = 3
low_threshold = 20.0

def calcRect(contours):
    r = cv.boundingRect(contours[0])
    rect = list(r)
    for cnt in contours[1:]:
        r = cv.boundingRect(cnt)
        if r[0] < rect[0]: rect[0] = r[0]
        if r[1] < rect[1]: rect[1] = r[1]
        if r[0]+r[2] > rect[0]+rect[2]: rect[2] = r[2]
        if r[1]+r[3] > rect[1]+rect[3]: rect[3] = r[3]
    rect = tuple(rect)
    return rect

def ImageClean(img, tresholdValue, debug = False):

    cv.bilateralFilter(img, 9, 90, 16)
    normal = ImageLevels(img, 30, 108, 10, False)
    sharpenForce = 2
    kernel = np.array([
        [0,-1*sharpenForce,0], 
        [-1*sharpenForce,(4*sharpenForce) + 1,-1*sharpenForce],
        [0,-1*sharpenForce,0]
    ])
    sharp = cv.filter2D(normal, -1, kernel)
    blur = cv.GaussianBlur(sharp, (9,9), 0)

    lower = np.array([0,   0,  218])
    upper = np.array([255, 16, 255])
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)

    # fill inner holes
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    des = cv.bitwise_not(mask)
    contours, hierarchy = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(des, [cnt], 0, 255, -1)

    kfactor = img.shape[0]/35
    kernel = np.ones((kfactor, kfactor), np.uint8)
    delate = cv.dilate(des, kernel, iterations = 4)

    bfactor = img.shape[0]/5
    if bfactor % 2 == 0: bfactor += 1
    blur = cv.GaussianBlur(delate, (bfactor, bfactor), 0)
    mask = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
    mask = np.array(mask, dtype=np.float)
    mask /= 255.0

    background = np.full(img.shape, (255,255,255), dtype=np.float)
    
    img = np.array(img, dtype=np.float)
    img = img * mask + background * (1.0 - mask)
    img = np.array(img, dtype=np.uint8)

    return img

def ImageCleanFalse2(img, tresholdValue, debug = False):

    '''
    cv.bilateralFilter(img, 9, 90, 16)
    blur = cv.GaussianBlur(img, (7,7), 0)

    lower = np.array([0,   0,  218])
    upper = np.array([255, 16, 255])
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)

    kernel = np.ones((4, 4), np.uint8)
    delate = cv.dilate(mask, kernel, iterations = 4)
    contours, hierarchy = cv.findContours(delate, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    rect = calcRect(contours)
    '''
    rect = (2, 2, img.shape[1]-4, img.shape[0]-4)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    img = ImageLevels(img, 10, 118, 10, True)
    sharpenForce = 1
    kernel = np.array([
        [0,-1*sharpenForce,0], 
        [-1*sharpenForce,(4*sharpenForce) + 1,-1*sharpenForce],
        [0,-1*sharpenForce,0]
    ])
    sharp = cv.filter2D(img, -1, kernel)
    cv.grabCut(sharp, mask, rect, bgdModel, fgdModel, 13, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img = img * mask2[:,:,np.newaxis]

    x,y,w,h = rect
    img = cv.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)

    return img
    

def ImageCleanFalse(img, tresholdValue, debug = False):

    cv.bilateralFilter(img, 9, 90, 16)
    blur = cv.GaussianBlur(img, (7,7), 0)

    lower = np.array([0,   0,  218])
    upper = np.array([255, 16, 255])
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)

    kernel = np.ones((4, 4), np.uint8)
    delate = cv.dilate(mask, kernel, iterations = 4)
    bgMask = cv.Canny(delate, 0, 50, apertureSize = 5)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(mask, kernel, iterations = 6)
    fgMask = cv.Canny(erosion, 0, 50, apertureSize = 5)

    #mask = np.zeros(img.shape[:2], np.uint8)
    #bgdModel = np.zeros((1,65), np.float64)
    #fgdModel = np.zeros((1,65), np.float64)
    #rect = (2, 2, img.shape[1]-4, img.shape[0]-4)
    #cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    mask = np.full(img.shape[:2], 2, dtype=np.uint8)
    mask[bgMask == 255] = 0
    mask[fgMask == 255] = 1
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    mask, bgdModel, fgdModel = cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img = img * mask2[:,:,np.newaxis]

    return img
    


