#!/usr/bin/python
# -*- coding: utf8 -*-

import math
import numpy as np
import scipy as sp
import cv2 as cv
import operator
from scipy import signal
from scipy import interpolate
from matplotlib import pyplot as plt

def ImageLevels(image, darkLevel = 0, middleLevel = 127, lightLevel = 0, autoLevel = True):

    extrem = [255] * 3
    if autoLevel:
        for i in range(3):
            h = cv.calcHist([image],[i],None,[256],[0,256])
            # Find peak for each channel
            extrem[i], _ = max(enumerate(h), key=operator.itemgetter(1))
        # Check for white background
        reset = False
        for a in extrem:
            for b in extrem:
                if math.fabs(a - b) > 100:
                    reset = True
                    break
            if a < 150: 
                reset = True
                break
        if reset:
            for i, _ in enumerate(extrem):
                extrem[i] = 255

    #print extrem
    #plt.hist(image.ravel(), 256, [0,256])
    #plt.show()
    #color = ('b','g','r')
    #for i,col in enumerate(color):
    #    histr = cv.calcHist([image],[i],None,[256],[0,256])
    #    plt.plot(histr, color = col)
    #    plt.xlim([0,256])
    #plt.show()

    # LEVELS
    levelsX = [[
        0, 
        darkLevel, 
        min(255, max(0, darkLevel + (point - lightLevel - darkLevel) / 2.)), 
        min(255, max(0, point - lightLevel)), 
        255
        ] for point in extrem]
    levelsY = [0, 0, 
               middleLevel, 
               255, 255]
    levels = [interpolate.interp1d(x, levelsY, kind='linear') for x in levelsX]

    levelsLut = [[[min(255, max(0, round(levels[j](i)))) for j in range(3)] for i in range(256)]]
    lut = np.asarray(levelsLut, dtype=np.uint8)
    image = cv.LUT(image, lut)

    return image



def ImageAutoLevels(image, autoLevelOfs = 0, middleLevel = 127):

    extrem = [0] * 3
    dark = [0] * 3
    for i in range(3):
        h = cv.calcHist([image],[i],None,[256],[0,256])
        
        medianHist = sp.signal.medfilt([v[0] for v in h], 5)

        # Find peak for each channel
        extrem[i], _ = max(enumerate(h), key=operator.itemgetter(1))

        for j, v in enumerate(medianHist[:120]):
            if v > 1:
                dark[i] = j
                break

    # LEVELS
    levelsX = [[
        0, 
        dark[0], 
        min(255, max(0, dark[0] + (ex + autoLevelOfs - dark[0]) / 2.)), 
        min(255, max(0, ex + autoLevelOfs)), 
        255
        ] for ex in extrem]
    levelsY = [0, 0, 
               middleLevel, 
               255, 255]
    levels = [interpolate.interp1d(x, levelsY, kind='linear') for x in levelsX]

    levelsLut = [[[min(255, max(0, round(levels[j](i)))) for j in range(3)] for i in range(256)]]
    lut = np.asarray(levelsLut, dtype=np.uint8)
    image = cv.LUT(image, lut)

    return image
