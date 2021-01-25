import numpy as np
import matplotlib.pyplot as plt
import argparse
from os import path
from scipy.io.wavfile import read, write
from numpy.fft import fft, ifft
import moviepy.editor as mp

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





#-------------------------------NEW----------------------------------





def get_args():
    print("get_args():".ljust(20), end="", flush=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str, default="", help="path to input video file")
    ap.add_argument("-l", "--limit", type=int, default=None, help="frame processing limit")

    args = vars(ap.parse_args())
    filename = args["audio"]
    limit = args["limit"]
    ifLim = (limit is not None)
    print("Complete.")
    return filename, limit, ifLim

def get_data(filename, ifLim=False, limit=0):
    #-------Extracting single channel audio wav file-------
    print("get_data():".ljust(20), end="", flush=True)
    stripFile, fileType = noFileType(filename)
    if (fileType == 'mp4'):
        if not (path.exists(stripFile +".wav")):
            #print("Extracting audio from video")
            clip = mp.VideoFileClip(filename)
            clip.audio.write_audiofile(stripFile + '.wav')

    sRate, data = read(stripFile + '.wav')
    if (len(data.shape) > 1):
        data = data[:, 0] #звук двухканальный, оставляем только один
    time = np.arange(0, len(data)/sRate, 1/sRate)
    data = signalNormalization(data, 0.95, 10000)
    if (ifLim and len(data)/sRate > limit):
        data = data[:sRate*limit]
    print("Complete.")
    return sRate, data

def get_spectrogram(sRate, data, frame_size, steps):
    #-------Calculating spectrum--------
    #print(sRate)
    frame_size = sRate//8
    timeStep = frame_size/sRate
    timeArray = np.arange(0, timeStep*steps, timeStep)
    dt = 1/sRate
    freq_vector = np.fft.rfftfreq(frame_size, d=dt)
    xlim = hzIndex(freq_vector, 2000)
    X = freq_vector[:xlim]
    spectrogram = np.zeros((steps, xlim))
    #print("frame_size = ", frame_size)
    #print("len)freq) = ", len(freq_vector))
    #print("steps = ", steps)
    #print("Applying fast fourier transform")
    for i in range(steps):
        print("get_spectrogram():".ljust(20) + str(i+1) + "/" + str(steps), end="\r")
        signal = data[i*frame_size:(i+1)*frame_size]
        Y = np.fft.rfft(signal)
        Y = np.abs(Y[:xlim])
        spectrogram[i] = Y
    print("get_spectrogram():".ljust(20) + "Complete.")
    return spectrogram, X, timeArray

def get_formants(spectrogram, X, steps):
    #--------Caluculating 1st formant------------
    #print("Calculating 1st formants")
    formantFreq = np.zeros(steps)
    formantID = np.zeros(steps)
    formantValue = np.zeros(steps)
    for i in range(steps):
        print("get_formants():".ljust(20) + str(i+1) + "/" + str(steps), end="\r")
        Y = spectrogram[i]
        stepX, stepTrueX, stepMax = steppingMax(X, Y, 200, 100)
        IDs, xmax, maxes = catchingMax(X, stepTrueX, 25, 1)
        freqID = IDs[np.argmax(maxes)]
        formantID[i] = freqID
        formantFreq[i] = xmax[np.argmax(maxes)]
        formantValue[i] = Y[freqID]
    print("get_formants():".ljust(20) + "Complete.")
    return formantID, formantFreq, formantValue

def get_good(data, steps, frame_size, spectrogram):
    #--------caluculating good frames----------------
    #print("Calculating good frames")
    signalMins = np.zeros(steps)
    SpectrumMaxes = np.zeros(steps)
    goodness = np.zeros(steps)
    for i in range(steps):
        print("get_good():".ljust(20) + str(i+1) + "/" + str(steps), end="\r")
        signal = data[i*frame_size:(i+1)*frame_size]
        Xsignal = np.arange(i*frame_size, (i+1)*frame_size)
        bendingX, bendingY = bendingMax(Xsignal, signal, 500, 100)
        smin = np.min(bendingY)
        smax = np.max(spectrogram[i])
        signalMins[i] = smin
        SpectrumMaxes[i] = smax
        if (smin > 1000 and smax > 2000000):
            goodness[i] = 1
    print("get_good():".ljust(20) + "Complete.")
    goodSum = 0
    for x in goodness:
        if x:
            goodSum += 1
            #print("Good frames: ", goodSum)
    return goodness, goodSum, signalMins, SpectrumMaxes

def plot_bend(data, steps, frame_size, signalMins, SpectrumMaxes):
    signal = data[:steps*frame_size]
    Xsignal = np.linspace(0, steps, steps*frame_size)
    plt.plot(Xsignal, np.abs(signal), '-', color=(0.0, 0.0, 1.0, 0.5))
    bendX, bendY = bendingMax(Xsignal, np.abs(signal), 200, 100)
    plt.plot(bendX, bendY, '.-', color=(1.0, 0.5, 0.5, 1.0))
    plt.show()

    plt.plot(np.linspace(0, steps, steps*frame_size), data[:steps*frame_size], '-', color=(0.0, 0.0, 1.0, 0.5))
    plt.plot([0, len(signalMins)], [0, 0], '-', color=(0.2, 0.2, 0.2))
    plt.plot(np.arange(0.5, len(signalMins)+0.5), signalMins, '.-', color=(1.0, 0.0, 0.0, 1.0))
    plt.plot([0, len(signalMins)], [1000, 1000], '-', color=(1.0, 0.0, 0.0, 0.5))
    plt.plot(np.arange(0.5, len(signalMins)+0.5), SpectrumMaxes/1000, '-', color=(0.0, 1.0, 0.0, 1.0))
    plt.plot([0, len(SpectrumMaxes)], [2000, 2000], '-', color=(0.0, 1.0, 0.0, 0.5))
    plt.grid(axis='x', linestyle='-')
    plt.xticks(np.arange(0, len(signalMins)))
    plt.show()

def plot_freq_response(formantFreq, formantValue, goodness):
    #-------frequency response--------------
    freqResponse = np.array([formantFreq, formantValue, goodness])
    goodFreqResponse = np.array([i for i in freqResponse.T if i[2] != 0])
    plt.plot(*goodFreqResponse.T[0:2], '.')
    plt.show()

def plot_spectrogram(timeArray, X, spectrogram, formantID):
    #------plotting spectrogram-----------
    fig, ax = plt.subplots(figsize=(20, 10))
    t = timeArray
    w = X
    xv, yv = np.meshgrid(t, w)
    z = spectrogram
    for i in range(len(z)):
        if( formantID[i] != 0 ):
            id = int(formantID[i])
            z[i][id] = -10000000
    z = z.T
    c = ax.pcolormesh(xv, yv, z, cmap="RdBu", vmin=-10000000, vmax=10000000)
    ax.axis([t.min(), t.max(), w.min(), w.max()])
    fig.colorbar(c, ax=ax)
    plt.show()

def postprocessing(spectrogram, X, data, frame_size):
    #-------Postprocessing--------------
    k = 0
    plotStart = 0
    plotEnd = 100
    for i in np.arange(plotStart, plotEnd):
        Y = np.abs(spectrogram[i])
        Ymax = np.amax(Y)
        maxIndex = np.argmax(Y)
        lines = np.array([X[maxIndex], X[maxIndex]])
        stepX, stepTrueX, stepMax = steppingMax(X, Y, 200, 100)
        IDs, xmax, maxs = catchingMax(X, stepTrueX, 25, 1)
        freq = xmax[np.argmax(maxs)]

        plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X, Y, "-", color=(0.7, 0.7, 0.7))
        plt.plot(X[maxIndex], Ymax, "g.")
        plt.ylim(0,10000000)
        plt.xlim(X[0], X[-1])
        plt.plot(stepX, stepMax, ".", color=(1.0, 0.7, 0.7))
        plt.plot(stepTrueX, stepMax, '.', color=(0.0, 0.5, 0.0))
        for j in range(1, 9):
            plt.plot([freq*j-25, freq*j, freq*j+25], [0,0,0], "r.-")

            plt.subplot(312)
            plt.plot(xmax, maxs)

            audRange = 4
            chunkStart = max(0, frame_size*(i-audRange))
            chunkEnd = min(len(data), frame_size*(i+1+audRange))
            frameStart = max(0, frame_size*i)
            frameEnd = min(len(data), frame_size*(i+1))
            dataX = np.arange(chunkStart,chunkEnd)
            dataChunk = data[chunkStart:chunkEnd]
            frameX = np.arange(frameStart, frameEnd)
            frameY = data[frameStart:frameEnd]
            bendingX, bendingY = bendingMax(frameX, frameY, 500, 100)
            minSignal = np.min(bendingY)
            plt.subplot(313)
            plt.plot([dataX[0], dataX[-1]], [0, 0], "-", color=(0.2, 0.2, 0.2))
            plt.plot(dataX, dataChunk)
            if (minSignal > 1000 and Ymax > 2000000):
                print("freq = ", freq)
                plt.title("good", color=(0.0, 1.0, 0.0))
            else:
                plt.title("bad", color=(1.0, 0.0, 0.0))
                plt.plot(bendingX, bendingY)
                plt.plot([frame_size*i, frame_size*(i+1)], [0,0])
                plt.show()
