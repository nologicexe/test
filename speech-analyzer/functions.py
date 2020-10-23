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
    print("Sorting...")
    sorted = np.sort(np.abs(signal))
    print("Sorting complete")
    normIndex = int(len(signal)*percentile)-1
    norm = np.abs(sorted[normIndex])
    multiplier = normValue/norm
    print("Multiplier = ", multiplier)
    print("Applying multiplier...")
    output = signal*multiplier
    print("Normalization complete.")
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
    #print("X = ", X)
    #print("Xmax = ", Xmax)
    for i in range(iR, len(X), iStep):
        sum = 0
        xSums.append(X[i])
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
    return xSums, sums

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
