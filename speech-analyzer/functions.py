import numpy as np
import matplotlib.pyplot as plt

def noFileType(filePath):
    splited = filePath.split(".")
    stripFile = ''
    for i in splited[:-2]:
        stripFile += i + '.'
    stripFile += splited[-2]
    fileType = splited[-1]
    return stripFile, fileType

def signalNormalization(signal, percentile, normValue):
    sorted = np.sort(np.abs(signal))
    normIndex = int(len(signal)*percentile)-1
    norm = np.abs(sorted[normIndex])
    multiplier = normValue/norm
    output = signal*multiplier
    return output

def hzIndex(X, hz):
    output = None
    for i in range(len(X)):
        if X[i] < hz:
            output = i
    return output


def steppingMax(X, Y, width, step):
    indexWidth = max(1, int(width//(X[1]-X[0])))
    indexStep = max(1, int(step//(X[1]-X[0])))
    #print("len(Y) = ", len(Y))
    #print("indexWidth = ", indexWidth)
    #print("indexStep = ", indexStep)
    rang = int( (len(Y)-indexWidth)//indexStep + 1)

    outY = np.zeros(rang)
    outTrueX = np.zeros(rang)
    outX = np.zeros(rang)

    for i in range(rang):
        start = indexStep*i
        end = min(len(Y), indexWidth+indexStep*i)
        #print("start, end = ", start, end)
        outY[i] = np.amax(Y[start:end])
        index = np.argmax(Y[start:end])+start
        outX[i] = X[start+indexWidth//2]
        outTrueX[i] = X[index]
    #plt.plot(X, Y, '-g')
    #plt.plot(outX, outY, '-r')
    #plt.plot(outTrueX, outY, '-b')
    #plt.show()
    return outX, outTrueX, outY

def catchingMax(X, Xmax, r, step):
    iR = max(1, int(r//(X[1]-X[0])))
    iStep = max(1, int(step//(X[1]-X[0])))
    #print("iR, iStep = ", iR, iStep)
    sums = [0]
    xSums = [0]
    IDs = [0]
    #print("X = ", X)
    #print("Xmax = ", Xmax)
    for i in range(iR, len(X), iStep):
        sum = 0
        xSums.append(X[i])
        IDs.append(i)
        for j in range(1, min(8, len(X)//i)):
            left = X[i*j-iR]
            right = X[i*j+iR]
            found = False
            for m in Xmax:
                if (left <= m <= right):
                    found = True
                    sum += 1
            if not found:
                sum -= 1
        sums.append(sum)
    sums = np.array(sums)
    #print(sums)
    #plt.subplot(212)
    #plt.plot(xSums, sums)
    #plt.show()
    return IDs, xSums, sums

def resonance(X, Y, n, r):
    hzStep = X[1]-X[0]
    indexRadius = r // hzStep
    Ymax = np.amax(Y)
    MaxIndex = np.argmax(Y)
    start = 0
    end = len(X)-1
    areaSum = np.zeros(n)
    areaX = np.arange(1, n+1)
    for i in range(1, n+1):
        f1X = MaxIndex / i
        if (X[int(f1X)] <= 60):
            areaSum[i-1] = 0
        else:
            for j in range(1, n+1):
                currentIndex = int(f1X*j)
                a = int(max(0, currentIndex-indexRadius))
                b = int(min(end+1, currentIndex+indexRadius+1))
                #print("a, b = ", a, b)
                areaSum[i-1] += sum(Y[a:b])
        #areaSum[i-1] /= i
    #plt.plot(areaX, areaSum)
    #plt.show()
    return areaX, areaSum

def bendingMax(X, Y, r, step):
    start = 0
    end = len(Y)
    n = 1+(end-1)//step
    outX = np.zeros(n)
    outY = np.zeros(n)
    for i in range(0, n):
        left = max(0, i*step - r)
        right = min(i*step+r+1, end)
        outX[i] = X[i*step]
        outY[i] = max(Y[left:right])
    return outX, outY


#----------needs review and evaluation------------
def countPicks(a, threshold):
    picks = 0
    risen = False
    if (a[0] > threshold):
        bigger = True
    else: bigger = False
    for i in a:
        if (i < threshold and bigger and risen):
            picks+=1
            bigger = False
        elif (i < threshold):
            bigger = False
        elif (i > threshold and not bigger):
            bigger = True
            risen = True
        elif (i > threshold):
            bigger = True
    return picks

def goodness(picks, min, max):
    if (min <= picks <= max):
        good = True
    else:
        good = False
    return good

def removeRepeat(a):
    output = []
    prev = a[0]
    output.append(prev)
    for next in a:
        #print(prev, next)
        if (next != prev):
            output.append(next)
            #print(output)
        prev = next
    return output
