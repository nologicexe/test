import numpy as np
import argparse
import time
import dlib
import cv2
from matplotlib import pyplot as plt
import functions as func
import time
import math
import os
from os import path
import json
#from fer import FER

def getBlinkDetection(filename, savefile=''):
    vid = cv2.VideoCapture(filename) #opening mp4 file

    vidWidth = int(vid.get(3))
    vidHeight = int(vid.get(4))
    ratio = vidWidth/vidHeight
    #vidWidth = 400
    #vidHeight = int(vidWidth/ratio)

    dFlag = args["d"]
    filePath = args["video"]
    outputName = func.getOutputName(filePath)
    print(filePath)
    if (vid.isOpened() == False):
        print("ERROR - cant open video file: ", args["video"])
        vid.release()
        return 0
    else:
        lenght = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarkData = np.zeros((lenght, 68, 2))
        facePosData = np.zeros((lenght, 5, 2))
        if not (path.exists(outputName+"_Data")):
            os.mkdir(outputName+"_Data")
        print(vidWidth, vidHeight)
        if (path.exists(outputName+"_Data/video.mp4")):
            os.remove(outputName+"_Data/video.mp4")
        print("before videowriter")
        out = cv2.VideoWriter(outputName+"_Data/video.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (vidWidth, vidHeight))
        print("after")
        for i in range(lenght):
            if i > 10000:
                break
            ret, frame = func.getFrame(vid) #, size=2000
            if not ret:
                break
            landmarkData[i] = func.landmarkFrame(frame)
            #print(landmarkData[i])
            facePosData[i] = func.facePos(landmarkData[i])
            func.drawLandmark(frame, landmarkData[i], showID=True)
            func.drawFacePos(frame, facePosData[i], landmarkData[i])
            # cv2.imshow("test", frame)
            # key = cv2.waitKey(1)
            # if (key == ord('q')):
            #    break
            func.printProgress("Processing frames:", i+1, lenght)
            out.write(frame)

        goodFrames = func.getGood(facePosData, r=5)
        blinkNormData = func.blinkNormData(landmarkData, goodFrames)
        #for i in range(len(blinkNormData)):
        #    print(blinkNormData[i][0]+blinkNormData[i][1], ",")
        #func.plotBlinkData(blinkNormData)
        blinkData = blinkNormData[:,0] + blinkNormData[:,1]
        for i in range(lenght):
            #print( goodFrames[i], blinkData[i])
            if (goodFrames[i] == 0):
                blinkData[i] == 0
        threshold, parsed, blinks, fastBlinks, closedEyesStart, closedEyesEnd = func.parse(blinkData, coeff=0.65)
        fastBlinks.sort()

        output = {
            "threshold": threshold,
            "parsed": parsed,
            "blinks": blinks,
            "fastBlinks": fastBlinks,
            "closedEyesStart": closedEyesStart,
            "closedEyesEnd": closedEyesEnd,
            "blinkData": blinkData.tolist(),
            "lenght": lenght
        }
        if (savefile != ''):
            with open(savefile, 'w') as json_file:
                json.dump(output, json_file)

        return output

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, default="", help="path to input video file")
    ap.add_argument('-d', action='store_true') #debugging mode (show video etc)
    args = vars(ap.parse_args())
    filename = args["video"]
    output = getBlinkDetection(filename, savefile='test.json')

    threshold = output["threshold"]
    parsed = output["parsed"]
    blinks = output["blinks"]
    fastBlinks = output["fastBlinks"]
    closedEyesStart = output["closedEyesStart"]
    closedEyesEnd = output["closedEyesEnd"]
    blinkData = np.array(output["blinkData"])
    lenght = output["lenght"]

    step = 5000
    overlap = 10

    start = 0
    end = step + overlap
    func.plotBlinks(0, range(0, min(end, lenght)), blinkData[start:min(end, lenght)], blinks, fastBlinks, closedEyesStart, closedEyesEnd, threshold=threshold)
    for i in range(1, lenght // step):
        start = i*step - overlap
        end = (i+1)*step + overlap
        func.plotBlinks(i, range(start, end), blinkData[start:end], blinks, fastBlinks, closedEyesStart, closedEyesEnd, threshold=threshold)
    if (lenght > step):
        start = (lenght//step)*step - overlap
        end = lenght
        func.plotBlinks(lenght//step+1, range(start, end), blinkData[start:end], blinks, fastBlinks, closedEyesStart, closedEyesEnd, threshold=threshold)
    #func.playback(frames, landmarkData, facePosData, goodFrames, blinkNormData, width=800)
    vid.release()
    cv2.destroyAllWindows()
