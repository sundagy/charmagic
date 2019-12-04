#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import math
import cv2 as cv

ratio = 3
kernel_size = 3
low_threshold = 20.0

def ImageClean(src, tresholdValue, debug = False):

    cv.bilateralFilter(src, 9, 90, 16)
    blur = cv.GaussianBlur(src, (7,7), 0)

    lower = np.array([0,   0,  218])
    upper = np.array([255, 16, 255])
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    
    #contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #for i, cnt in enumerate(contours):
    #    epsilon = 0.03*cv.arcLength(cnt, True)
    #    approx = cv.approxPolyDP(cnt, epsilon, True)
    #    contours[i] = approx
    #mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    #cv.drawContours(mask, contours, -1, (255, 0, 0), 2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    #kernel2 = np.ones((5, 5), np.uint8)
    #mask = cv.dilate(mask, kernel2, iterations = 2)

    #src = src.astype(float) / 255
    #mask = mask.astype(float) / 255
    #dst = cv.add(mask, src)
    #mask = np.array(mask)
    #src_rs = np.array(src)
    #dst = src_rs * (mask / 255)
    #img_blur = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)
    
    return mask
