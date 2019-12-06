#!/usr/bin/python
# -*- coding: utf8 -*-

import math
import numpy as np
import time
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import interpolate

def ImageHUE(image, hue = 0, saturation = 0, debug = False):
    # HUE shift
    if hue != 0 or saturation != 0:
        destHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        satF = float(saturation + 127) / 127.0
        lut = [
                [
                    [
                        (i + hue) % 180, 
                        max(0, min(255, i * satF)), 
                        i
                    ] for i in range(256)
                ]
            ]
        lutBytes = np.asarray(lut, dtype = np.uint8)
        destHSV = cv.LUT(destHSV, lutBytes)
        image = cv.cvtColor(destHSV, cv.COLOR_HSV2BGR)

    if debug:
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv.calcHist([image],[i],None,[256],[0,256])
            plt.plot(histr, color = col)
            plt.xlim([0,256])
        #plt.ion()
        plt.show()
        #time.sleep(1)
        #plt.close('all')
        #plt.ioff()

        dest_sm = cv.resize(image, None, fx=0.06, fy=0.06, interpolation=cv.INTER_CUBIC)
        hsv = destHSV = cv.cvtColor(dest_sm, cv.COLOR_BGR2HSV)

        pixel_colors = dest_sm.reshape((np.shape(dest_sm)[0]*np.shape(dest_sm)[1], 3))
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()

        h, s, v = cv.split(hsv)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
        plt.show()

    return image
