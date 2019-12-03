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

    cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
    cv.imshow("image", dest)
    cv.waitKey(500)

    if not os.path.exists(os.path.dirname(destname)):
        os.makedirs(os.path.dirname(destname))

    distname, _ = os.path.splitext(destname)
    cv.imwrite(distname + ".png", dest, [int(cv.IMWRITE_PNG_COMPRESSION), 100])

cv.namedWindow("image", cv.WINDOW_AUTOSIZE)

base = 'P 242'
dest = 'OUTPUT'
for filename in os.listdir(base):
    if filename.endswith(".JPG"):
        process(os.path.join(base, filename), os.path.join(dest, filename))

