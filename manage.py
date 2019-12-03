#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import scipy as sp
import os
import math
import time
import cv2 as cv
from argparse import ArgumentParser
from lib_crop import ImageCrop
from lib_levels import ImageLevels
from lib_hue import ImageHUE
from lib_clean import ImageClean
from threading import Timer

basePath = 'PHOTOS'
destPath = 'OUTPUT'
TILES_COL = 6
PREVIEW_SIZE = 140
srcImages = []
srcFilenames = []
updateTimer = None
darkLevel = 0
middleLevel = 127
lightLevel = 0
saturation = 127
hue = 127
windowName = "image"
mainTiles = cv.vconcat([])
enablePreview = 1
enableAutolevel = 1

parser = ArgumentParser()
parser.add_argument("-f", "--folder", dest="folder", default="", help="Custom folder to process")
parser.add_argument("-d", "--debug", dest="debug", default=False, action="store_true", help="Output debug info")
args = parser.parse_args()

def concatTiles(im_list_2d):
    return cv.vconcat([cv.hconcat(im_list_h) for im_list_h in im_list_2d])

def updateTilesImpl():
    global srcImages
    global darkLevel
    global middleLevel
    global lightLevel
    global hue
    global saturation
    global mainTiles
    global enablePreview
    global enableAutolevel
    
    nrows = []
    for row in srcImages:
        nrow = []
        for img in row:
            if enablePreview == 1:
                dest = ImageLevels(img, darkLevel, middleLevel, lightLevel, enableAutolevel == 1)
                dest = ImageHUE(dest, hue - 127, saturation - 127)
                nrow.append(dest)
            else:
                nrow.append(img)
        nrows.append(nrow)

    mainTiles = concatTiles(nrows)

def updateTiles():
    global updateTimer
    if updateTimer != None:
        updateTimer.cancel()
    updateTimer = Timer(1.0, updateTilesImpl)
    updateTimer.start()

def onDark(val):
    global darkLevel
    darkLevel = val
    updateTiles()
    
def onLight(val):
    global lightLevel
    lightLevel = val
    updateTiles()
    
def onMiddle(val):
    global middleLevel
    middleLevel = val
    updateTiles()
    
def onSaturation(val):
    global saturation
    saturation = val
    updateTiles()
    
def onHUE(val):
    global hue
    hue = val
    updateTiles()
    
def onPreview(val):
    global enablePreview
    enablePreview = val
    updateTiles()

def onAutolevel(val):
    global enableAutolevel
    enableAutolevel = val
    updateTiles()

for folderIdx, folder in enumerate(os.listdir(basePath)):
    if args.folder != "" and args.folder != folder:
        continue

    print "Scan folder", folder

    if os.path.isdir(os.path.join(destPath, folder)): 
        print folder, "is already exist, SKIP"
        continue
    
    srcImages = []
    srcFilenames = []

    row = []
    files = os.listdir(os.path.join(basePath, folder))
    files = [fn for fn in files if fn.lower().endswith(".jpg")]
    
    if len(files) == 0: 
        continue

    for idx, filename in enumerate(files):
        path = os.path.join(folder, filename)
        srcFilenames.append(path)

        fn = os.path.join(basePath, path)
        image = cv.imread(fn)
        image = ImageCrop(image, args.debug)
        image = ImageClean(image, args.debug)
        image = cv.resize(image, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation = cv.INTER_AREA)
        row.append(image)
        if len(row) == TILES_COL:
            srcImages.append(row)
            row = []
        print "Read", idx+1, "/", len(files)

    #fill tail for tiles
    if len(row) > 0:
        for i in range(len(row), TILES_COL):
            blank = np.zeros((PREVIEW_SIZE,PREVIEW_SIZE,3), np.uint8)
            blank[:, 0:PREVIEW_SIZE] = (255,255,255)
            row.append(blank)
        srcImages.append(row)
        row = []

    cv.namedWindow(windowName, cv.WINDOW_AUTOSIZE)
    cv.setWindowTitle(windowName, folder)
    cv.createTrackbar("Preview", windowName, enablePreview, 1, onPreview)
    cv.createTrackbar("AutoLevel", windowName, enableAutolevel, 1, onAutolevel)
    cv.createTrackbar("Dark", windowName, darkLevel, 255, onDark)
    cv.createTrackbar("Middle", windowName, middleLevel, 255, onMiddle)
    cv.createTrackbar("Light", windowName, lightLevel, 255, onLight)
    cv.createTrackbar("HUE", windowName, hue, 255, onHUE)
    cv.createTrackbar("Saturation", windowName, saturation, 255, onSaturation)
    
    mainTiles = concatTiles(srcImages)
    updateTiles()
    print "Press Enter to Accept changes, ESC to exit"

    while True:
        cv.imshow(windowName, mainTiles)
        keyCode = cv.waitKey(100)
        if keyCode != -1:
            break
    updateTimer.cancel()
    cv.destroyWindow(windowName)

    if keyCode == 27:
        break

    if keyCode == 13:
        for idx, path in enumerate(srcFilenames):
            print "Write", idx + 1, "/", len(srcFilenames)

            fn = os.path.join(basePath, path)
            destfn = os.path.join(destPath, path)
            destfn, _ = os.path.splitext(destfn)

            if not os.path.exists(os.path.dirname(destfn)):
                os.makedirs(os.path.dirname(destfn))

            image = cv.imread(fn)
            image = ImageCrop(image)
            image = ImageLevels(image, darkLevel, middleLevel, lightLevel)
            image = ImageHUE(image, hue - 127, saturation - 127)
            #cv.imwrite(destfn + ".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), 100, int(cv.IMWRITE_JPEG_PROGRESSIVE), 1])
            cv.imwrite(destfn + ".png", image, [int(cv.IMWRITE_PNG_COMPRESSION), 100])
    else:
        print "Canceled"
