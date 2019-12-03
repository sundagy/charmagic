#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import scipy as sp
import random as rng
import os
import math
import time
import cv2 as cv
import operator
from scipy import signal
from scipy import interpolate
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

parser = ArgumentParser()
parser.add_argument("-f", "--folder", dest="folder", default="", help="Custom folder to process")
parser.add_argument("-d", "--debug", dest="debug", default=False, action="store_true", help="Output debug info")
args = parser.parse_args()

rng.seed(12345)
max_lowThreshold = 100
ratio = 3
kernel_size = 3
low_threshold = 20.0

def crop(src):
    src_rs = cv.resize(src, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)
    src_gray = cv.cvtColor(src_rs, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(src_gray, (5, 5))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    contours, _ = cv.findContours(detected_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter too small objects
    contours = [i for i in contours if cv.arcLength(i, False) > 15]
    if len(contours) == 0:
        return

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 1, True) # Approximates a polygonal curve(s) with the specified precision.
        boundRect[i] = cv.boundingRect(contours_poly[i]) # Calc rect

    if args.debug:
        drawing = np.zeros((detected_edges.shape[0], detected_edges.shape[1], 3), dtype=np.uint8)
        cv.drawContours(drawing, contours_poly, -1, (255, 255, 255), 2)
        cv.imshow("image", img_blur)
        cv.waitKey(500)
        cv.imshow("image", detected_edges)
        cv.waitKey(500)
        cv.imshow("image", drawing)
        cv.waitKey(500)

    rect = list(boundRect[0])
    for r in boundRect[1:]:
        if r[0] < rect[0]:
            rect[2] += rect[0]-r[0]
            rect[0] = r[0]
        if r[1] < rect[1]:
            rect[3] += rect[1] - r[1]
            rect[1] = r[1]
        if r[0]+r[2] > rect[0]+rect[2]:
            rect[2] = r[0]+r[2] - rect[0]
        if r[3] > rect[3]:
            rect[3] = r[1]+r[3] - rect[1]
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    # correct ratio
    factor = 1200. / 900.
    fact_act = float(w) / float(h)
    if fact_act > factor:
        hgap = float(w)*(1/factor) - float(h)
        y -= int(math.ceil(hgap / 2.))
        h += int(math.ceil(hgap))
    else:
        wgap = float(h)*factor - float(w)
        x -= int(math.ceil(wgap / 2.))
        w += int(math.ceil(wgap))

    # add padding
    xp = int(math.ceil(float(w) * 0.1))
    yp = int(math.ceil(float(h) * 0.1))
    x -= xp
    w += xp * 2
    y -= yp
    h += yp * 2
    
    if y < 0: y = 0
    if x < 0: x = 0

    x *= 5
    y *= 5
    w *= 5
    h *= 5
    src = src[y:y + h, x:x + w]

    return cv.resize(src, (1200, 900), interpolation=cv.INTER_CUBIC)

def process(filename, destname, autoLevelOfs = 0, middleLevel = 1.0, hue = 0, saturation = 0):
    src = cv.imread(filename)
    dest = crop(src)

    hist = []
    extrem = [0] * 3
    dark = [0] * 3
    for i in range(3):
        h = cv.calcHist([dest],[i],None,[256],[0,256])
        hist.append(h)
        
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
        dark[0] + (ex + autoLevelOfs - dark[0]) / 2., 
        min(255, max(0, ex + autoLevelOfs)), 
        255
        ] for ex in extrem]
    levelsY = [0, 0, math.floor(middleLevel * 127), 255, 255]
    levels = [interpolate.interp1d(x, levelsY, kind='linear') for x in levelsX]

    levelsLut = [[[min(255, max(0, round(levels[j](i)))) for j in range(3)] for i in range(256)]]
    lut = np.asarray(levelsLut, dtype=np.uint8)
    dest = cv.LUT(dest, lut)

    # HUE shift
    if hue != 0 or saturation != 0:
        destHSV = cv.cvtColor(dest, cv.COLOR_BGR2HSV)
        y = [[ (
            math.floor(i + (hue / 2.)) % 180, 
            math.floor(min(255, max(0, i + (255 * (saturation / 100.0)) ))), 
            i 
                ) for i in range(256)]]
        lut = np.asarray(y, dtype = np.uint8)
        destHSV = cv.LUT(destHSV, lut)
        dest = cv.cvtColor(destHSV, cv.COLOR_HSV2BGR)

    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", dest)
    cv.waitKey(500)

    if args.debug:
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv.calcHist([dest],[i],None,[256],[0,256])
            plt.plot(histr, color = col)
            plt.xlim([0,256])
        #plt.ion()
        plt.show()
        #time.sleep(1)
        #plt.close('all')
        #plt.ioff()

        dest_sm = cv.resize(dest, None, fx=0.06, fy=0.06, interpolation=cv.INTER_CUBIC)
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

    if not os.path.exists(os.path.dirname(destname)):
        os.makedirs(os.path.dirname(destname))
   
    #cv.imwrite(destname, dest, [int(cv.IMWRITE_JPEG_QUALITY), 100, int(cv.IMWRITE_JPEG_PROGRESSIVE), 1])
    distname, _ = os.path.splitext(destname)
    cv.imwrite(distname + ".png", dest, [int(cv.IMWRITE_PNG_COMPRESSION), 100])

cv.namedWindow("image", cv.WINDOW_AUTOSIZE)

base = 'PHOTOS'
dest = 'OUTPUT'
for folder in os.listdir(base):
    if args.folder != "" and args.folder != folder:
        continue
    for filename in os.listdir(os.path.join(base, folder)):
        if filename.lower().endswith(".jpg"):
            process(os.path.join(base, folder, filename), 
                    os.path.join(dest, folder, filename),
                    autoLevelOfs=0,
                    middleLevel=1,
                    hue=0,
                    saturation=0)
            break




