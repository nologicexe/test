import time
import dlib
import cv2
import numpy as np
import copy
#from fer import FER
from matplotlib import pyplot as plt
from numpy import linalg
import os
from os import path

def getOutputName(inputName):
  buff = inputName.split('/')
  fileName = buff[-1].split('.')[0]
  outputName = ""
  for item in buff[:-1]:
    outputName += item + "/"
  outputName += fileName
  return outputName

#-------------------saving----------------------


def printProgress(taskName, curr, lenght):
    if(curr == lenght):
        print(taskName, ": ", curr, "/", lenght)
    else:
        print(taskName, ": ", curr, "/", lenght, end="\r")

def saveArray(array, dirName, fileName):
    if not(path.exists(dirName)):
        os.mkdir(dirName)
    file = open(dirName+"/"+fileName, "wb")
    np.save(file, array)
    file.close

def loadArray(fileName):
    file = open(fileName, "rb")
    array = np.load(file, allow_pickle=True)
    file.close
    return array

#--------------------Frames--------------------


def resize(image, width, rotated=False, inter=cv2.INTER_AREA):
    (w,h) = (image.shape[1], image.shape[0])
    ratio = width/float(w)
    height = float(h)*ratio
    #print(width, height)
    dim = (int(width), int(height))
    if (rotated):
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def testGetFrames(lenght, i):
    return [i]*lenght

def getFrame(video, size=400):
    ret, frame = video.read()
    if (ret == True):
        #printProgress("Loading video", i, lenght)
        if(size is not None):
            frame = resize(frame, size, False)
    return ret, frame

def getFrames(video, lenght=None, size=None, rotated=False):
    if(lenght == None):
        lenght = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    i = 0
    while (True):
        ret, frame = video.read()
        if (ret == True):
            i += 1
            if (i > lenght):
                break
            #printProgress("Loading video", i, lenght)
            if(size is not None):
                frame = resize(frame, size, rotated)
            frames.append(frame)
        else:
            break
    return frames

def convert2vid(folder):
    print("temp")


#-------------------------Landmark----------------------------


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def norm(a):
    return np.sqrt(a[0]**2+a[1]**2)

def landmarkFrame(frame, faceDetector=dlib.get_frontal_face_detector(), shapePredictor=dlib.shape_predictor("dataset.dat")):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = faceDetector(gray, 1)
    if (len(rects) > 0):
        shape = shapePredictor(gray, rects[0])
        landmarkData = np.array(shape_to_np(shape))
    else:
        landmarkData = np.zeros((68, 2))
    #print(landmarkData)
    return landmarkData

def landmark(frames, faceDetector=dlib.get_frontal_face_detector(), shapePredictor=dlib.shape_predictor("dataset.dat")):
    landmarkData = []
    lenght = len(frames)
    i = 0
    for frame in frames:
        #cv2.imshow("test", frame)
        #cv2.waitKey(0)
        i += 1
        #printProgress("Landmarking", i, lenght)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = faceDetector(gray, 1)
        if (len(rects) > 0):
            shape = shapePredictor(gray, rects[0])
            shape = shape_to_np(shape)
            landmarkData.append(shape)
        else:
            landmarkData.append(np.zeros((68, 2)))
    landmarkData = np.array(landmarkData)
    #print(landmarkData)
    return landmarkData


#-------------------------face position----------------------------------


def facePos(landmark):
    if landmark[0][0] == 0:
        facePos = np.zeros((5, 2))
    else:
        #print(len(landmark))
        A, B, C, D, N = np.array(landmark[36]), np.array(landmark[45]), np.array(landmark[48]), np.array(landmark[54]), np.array(landmark[30])
        #print(A, B, C, D, N)
        S = (A+B)/2
        P = (C+D)/2
        Q = (A+C)/2
        R = (B+D)/2
        L = norm(C-P)
        H = norm(S-P)/2
        a = np.cross(S-N, S-P)/norm(S-P)
        h = np.cross(Q-N, Q-R)/norm(Q-R)
        xcoef = np.abs(a/L)
        ycoef = np.abs(h/H)

        facePos = [np.array([xcoef, ycoef]), S.astype(int), P.astype(int), Q.astype(int), R.astype(int)]
        #print(facePos == None)
    #print(facePos)
    return facePos


#-------------------------------------goood frames-----------------------------------------


def getGood(facePosData, r=15):
    goodFrames = np.zeros(len(facePosData))
    num = 0
    for i in range(r, len(facePosData)-r):
        printProgress("Calculating good frames", i, len(facePosData)-r-1)
        flag = True
        for j in range(-r, r+1):
            if (facePosData[i + j][0, 0] == 0):
                flag = False
            else:
                coef = facePosData[i+j][0]
                if (coef[0] >  1.0 or coef[1] > 0.6):
                    flag = False
        if flag:
            goodFrames[i] = 1
            num += 1
    print("Good frames: ", num, "/", len(facePosData))
    return goodFrames


#----------------------------------norm------------------------------


def scalar(a, b):
    return a[0]*b[0]+a[1]*b[1]

def negativeNorm(a):
    if a[1] < 0:
        return -norm(a)
    else:
        return norm(a)

def dotProject(x, l1, l2):
    h = scalar(l2-l1, x-l1)/norm(l2-l1)
    vec = h/norm(l2-l1)*(l2-l1)
    dot = l1 + vec
    return dot

#distance from point to line
def distance(x, l1, l2):
    dot = dotProject(x, l1, l2)
    return negativeNorm(x-dot)

#sum of 2 distances from points on opposite side of the line
def fullDistance(x1, y1, l1, l2):
    return distance(x1, l1, l2) - distance(y1, l1, l2)

#sum of 2 distances from points on one side of the line
def dualDistance(x1, x2, l1, l2):
    return distance(x1, l1, l2) + distance(x2, l1, l2)

#sum of 2 full distances
def dualFullDistance(l1, x1, x2, l2, y1, y2):
    return fullDistance(x1, y1, l1, l2) + fullDistance(x2, y2, l1, l2)

def getDualDistance(l1, x1, x2, l2, y1, y2):
    top = dualDistance(x1, x2, l1, l2)
    bot = dualDistance(y1, y2, l1, l2)
    return [top, bot, top-bot]

def getEyeWidth(l1, x1, x2, l2, y1, y2):
    return norm(l2-l1)

def blinkNormData(landmarkData, goodFrames, countGood=True):
    norm = []
    for i in range(len(landmarkData)):
        landmark = landmarkData[i]
        if (landmark[0, 0] == 0 or (countGood and goodFrames[i] == 0)):
            norm.append(np.array([0, 0]))
        else:
            for x in landmark:
                x[1] *= -1
            width = np.array([getEyeWidth(*landmark[36:42]), getEyeWidth(*landmark[42:48])])
            full = np.array([dualFullDistance(*landmark[36:42]), dualFullDistance(*landmark[42:48])])
            norm.append(full/width)
    norm = np.array(norm)
    return norm

def getFastBlinks(blinks, fps=30, fastTime=1.0):
  fastBlinks = set()
  for i in range(1, len(blinks)-1):
    if ((blinks[i+1]-blinks[i-1])/fps <= fastTime):
      #print("i:", i-1, i, i+1)
      #print("blinks:", blinks[i-1:i+2])
      fastBlinks.update(blinks[i-1:i+2])
  fastBlinks = list(fastBlinks)
  #print(fastBlinks)
  return fastBlinks

def getBlinks(a, indexZero, threshold=0.8, fps=30, closeTime=0.3, fastTime=1.0):
  blinked = None
  start, end = 0, 0
  blinks = []
  closedEyesStart = []
  closedEyesEnd = []
  for i in range(len(a)):
    if (blinked is None and a[i] > threshold):
      blinked = False
    if ((a[i] <= threshold) and (blinked is not None) and (not blinked)):
      blinked = True
      start = i
    if (a[i] > threshold and blinked):
      blinked = False
      end = i-1
      if((end-start)/fps <= closeTime):
        blinks.append((end+start)/2 + indexZero)
      else:
        closedEyesStart.append(start+indexZero)
        closedEyesEnd.append(end+indexZero)
  fastBlinks = getFastBlinks(blinks, fps, fastTime)
  return blinks, fastBlinks, closedEyesStart, closedEyesEnd

def parse(a, coeff=0.5, threshold=0.8, fps=30, closeTime=0.3, fastTime=1.0):
    blinks = []
    fastBlinks = []
    closedEyesStart = []
    closedEyesEnd = []
    started = False
    start = 0
    end = 0
    parsedRanges = []
    for i in range(len(a)):
        if(a[i] != 0 and not started):
            started = True
            start = i
        if(a[i] == 0 and started):
            started = False
            end = i
            parsedRanges.append([start,end])
    max = np.amax(a)
    min = np.amin(a)
    threshold = min + coeff*(max-min)
#    print(threshold)
    for start,end in parsedRanges:
        lenght = len(a[start:end])
        blinksBuff, fastBlinksBuff, closedEyesStartBuff, closedEyesEndBuff = getBlinks(a[start:end], start, threshold)
        blinks.extend(blinksBuff)
        fastBlinks.extend(fastBlinksBuff)
        closedEyesStart.extend(closedEyesStartBuff)
        closedEyesEnd.extend(closedEyesEndBuff)
    #    print("---", "\n", blinks, "\n", fastBlinks, "\n", closedEyes)
  #  plot(a[start:end], b=blinks, c=fastBlinks, d = closedEyes, threshold=threshold)
    return threshold, parsedRanges, blinks, fastBlinks, closedEyesStart, closedEyesEnd


#-------------------------plotting----------------------------------------


def getPlotSize(H, w, frames):
    W = frames*4
    k = W/w
    h = H/k
    dpi1 = 100
    h /= dpi1
    w /= dpi1
    dpi2 = dpi1 * k
    return h, w, dpi1, dpi2

def parseBlinks(list, min, max):
    if (len(list) == 0):
        return list
    else:
        start = -1
        end = -1
        for i in range(len(list)):
            if (list[i] < min):
                start = i
            if (list[i] <= max):
                end = i
        start += 1
        if (end == -1):
            return []
        end += 1
        return list[start:end]

def plotBlinks(num, x, y, blinks, fastBlinks, closedEyesStart, closedEyesEnd, threshold=0.8, savename="test"):
    H=1000
    blinks = parseBlinks(blinks, x[0], x[-1])
    fastBlinks = parseBlinks(fastBlinks, x[0], x[-1])
    closedEyesStart = parseBlinks(closedEyesStart, x[0], x[-1])
    closedEyesEnd = parseBlinks(closedEyesEnd, x[0], x[-1])
    #print(blinks, "\n", fastBlinks, "\n", closedEyesStart, "\n", closedEyesEnd)
    h, w, dpi1, dpi2 = getPlotSize(H, 1000, len(y))
    smol=h/H*dpi1*1
    fine=smol*2
    thicc=smol*5

    fig = plt.figure(figsize=(w, h), dpi=dpi1)
    ax = fig.add_subplot(1, 1, 1)
    major_xticks=np.arange(x[0]-10, x[-1]+10, 60)
    minor_xticks=np.arange(x[0]-10, x[-1]+10, 3)
    major_yticks=np.arange(0, 2.5, 0.5)
    minor_yticks=np.arange(0, 2.5, 0.1)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor = True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor = True)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)

    ax.plot(x, y, color=(1.0, 0, 0, 0.3), linestyle="-", marker='.', linewidth=smol, markersize=fine)
    ax.plot(x, y, color=(1.0, 0, 0, 1.0), linestyle="", marker='.', linewidth=smol, markersize=fine)
    x1 = [threshold]*len(blinks)
    x2 = [threshold]*len(fastBlinks)
    x3 = [threshold]*len(closedEyesStart)
    x4 = [threshold]*len(closedEyesEnd)
    #print("--plot--", "\n", b, "\n", c, "\n", d, "\n", x1, "\n", x2, "\n", x3)
    ax.plot(blinks, x1, color=(0, 0, 1.0, 1.0), linestyle="", marker='.', linewidth=smol, markersize=thicc, label="blinks")
    ax.plot(fastBlinks, x2, color=(0, 1.0, 0, 1.0), linestyle="", marker='.', linewidth=smol, markersize=thicc, label="fast blinks (> 3 times/second)")
    if (len(closedEyesStart) > 0):
        closedEyesStart = np.array(closedEyesStart)
        closedEyesEnd = np.array(closedEyesEnd)
        ax.plot(closedEyesStart, x3, color=(0, 1.0, 1.0, 1.0), linestyle="", marker='>', linewidth=smol, markersize=thicc, label="closed eyes > 0.3 sec")
        ax.plot(closedEyesEnd, x4, color=(0, 1.0, 1.0, 1.0), linestyle="", marker='<', linewidth=smol, markersize=thicc)
    #plt.legend()
    #print("dpi2=", dpi2)
    plt.savefig(savename+str(num)+".png", dpi=dpi2)
    plt.show()

def plotBlinkData(blinkNormData):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    major_xticks=np.arange(0, 400, 60)
    minor_xticks=np.arange(0, 400, 3)
    major_yticks=np.arange(0, 1.0, 0.5)
    minor_yticks=np.arange(0, 1.0, 0.1)
    ax.set_xticks(major_xticks)
    ax.set_xticks(minor_xticks, minor = True)
    ax.set_yticks(major_yticks)
    ax.set_yticks(minor_yticks, minor = True)
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)
    ax.plot(blinkNormData[:,0], color=(1.0, 0, 0, 0.5), linestyle="-", marker='.', linewidth=1, markersize=2)
    ax.plot(blinkNormData[:,1], color=(0, 0, 1.0, 0.5), linestyle="-", marker='.', linewidth=1, markersize=2)
    plt.show()

#----------------------video-----------------------------


def drawLandmark(frame, landmark, showID=False):
    if(landmark[0, 0] != 0):
        #cv2.putText(frame, "not None", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 0)
        i = 0
        for (x,y) in landmark:
            x = int(x)
            y = int(y)
            #print(x, y)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            if showID:
                cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 0)
            i += 1

def drawFacePos(frame, facePos, landmark):
    if(facePos[0,0] != 0):
        coef, S, P, Q, R = facePos
        #print(facePos)
        if (coef[0] > 1.2 or coef[1] > 0.6):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.line(frame, tuple(S.astype(int)), tuple(P.astype(int)), color, 2)
        cv2.line(frame, tuple(Q.astype(int)), tuple(R.astype(int)), color, 2)
        cv2.putText(frame, str(coef[0]), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 0)
        cv2.putText(frame, str(coef[1]), (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 0)
        A, B, C, D, N = landmark[48].astype(int), landmark[54].astype(int), landmark[36].astype(int),landmark[45].astype(int), landmark[30].astype(int)
        cv2.circle(frame, tuple([N[0], N[1]]), 2, (255, 0, 0), -1)
        cv2.circle(frame, tuple([A[0], A[1]]), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple([B[0], B[1]]), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple([C[0], C[1]]), 2, (0, 255, 0), -1)
        cv2.circle(frame, tuple([D[0], D[1]]), 2, (0, 255, 0), -1)

def drawGoodFrames(frame, goodFrame):
    if (goodFrame == 1):
        cv2.putText(frame, "GOOD", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 0)
    else:
        cv2.putText(frame, "BAD", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 0)

def drawBlinkNormData(frame, blinkNorm):
    cv2.putText(frame, str(blinkNorm[0]), (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 0)
    cv2.putText(frame, str(blinkNorm[0]), (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 0)

def playback(frames, landmarkData, facePosData, goodFrames, blinkNormData, drawingLandmark=True,
                drawingFacePos=True, drawingGoodFrames=True, drawingBlinkNormData=True, name="test", fps=30, width=None):
    fpsMultiplier = 1.0
    size = frames[0].shape[:2]
    curFps = 0
    fCount = 0
    t = time.time()
    i = 0
    fpsFreq = 2.0
    reversed = False
    paused = False
    ended = False
    while True:
        if (((i == len(frames) - 1) and (not reversed)) or ((i == 0) and reversed)):
            ended = True
            paused = True
        #current fps
        if (time.time() - t >= 1.0/fpsFreq):
            if paused:
                curFps = 0
            else:
                curFps = int(fCount*fpsFreq)
            fCount = 0
            t = time.time()
        fCount += 1
        frame = copy.deepcopy(frames[i])
        if drawingLandmark:
            drawLandmark(frame, landmarkData[i])
        if drawingFacePos:
            drawFacePos(frame, facePosData[i], landmarkData[i])
        if drawingGoodFrames:
            drawGoodFrames(frame, goodFrames[i])
        if drawingBlinkNormData:
            drawBlinkNormData(frame, blinkNormData[i])
        cv2.putText(frame, str(curFps) + "(" + str(fps*fpsMultiplier) + ")", (20,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, 0)
        cv2.putText(frame, str(i) + "/" + str(len(frames) - 1), (20, size[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 0)
        #displaying
        if (width is not None):
            frame = resize(frame, width)
        cv2.imshow(name, frame)
        key = cv2.waitKey(int(1000/(fps*fpsMultiplier)))

        if (paused):
            i += 0
        else:
            if (reversed):
                i -= 1
            else:
                i += 1

        if (key == ord('q')):
            break
        if (key == ord('w')):
            fpsMultiplier *= 2
        if (key == ord('s')):
            fpsMultiplier /= 2
        if (key == ord('a')):
            if i > 0:
                i -= 1
        if (key == ord('d')):
            if i < len(frames)-1:
                i += 1
        if (key == ord('r')):
            i = 0
            reversed = False
            paused = False
            ended = False
        if (key == ord('e')):
            reversed = not reversed
        if (key == ord('p')):
            if (not ended):
                paused = not paused
