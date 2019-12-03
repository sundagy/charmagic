#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import math
import cv2 as cv

ratio = 3
kernel_size = 3
low_threshold = 20.0

def ImageClean(src, debug = False):
    
    src_rs = cv.resize(src, None, fx=0.2, fy=0.2, interpolation=cv.INTER_CUBIC)
    src_gray = cv.cvtColor(src_rs, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(src_gray, (10, 10))

    _, mask = cv.threshold(img_blur, 2, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    mask = np.array(mask)
    src_rs = np.array(src)
    
    dst = src_rs * (mask / 255)
    
    #img_blur = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)
    
    return dst
