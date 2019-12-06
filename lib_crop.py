#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import math
import cv2 as cv

ratio = 3
kernel_size = 3
low_threshold = 20.0

def ImageCrop(src, debug = False):
    src_rs = cv.resize(src, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)

    img_blur = cv.GaussianBlur(src_rs, (7,7), 0)
    lower = np.array([0,   0,  218])
    upper = np.array([255, 16, 255])
    hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
    src_gray = cv.inRange(hsv, lower, upper)

    #src_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    detected_edges = cv.Canny(src_gray, low_threshold, low_threshold * ratio, kernel_size)
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

    if debug:
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
    xp = int(math.ceil(float(w) * 0.2))
    yp = int(math.ceil(float(h) * 0.2))
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
