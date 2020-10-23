import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write
from numpy.fft import fft, ifft
import argparse
import moviepy.editor as mp
import cv2
import scipy
from os import path
import math
import functions as func

ap = argparse.ArgumentParser()
ap.add_argument("audio", type=str, default="", help="path to input video file")
#ap.add_argument("")
args = vars(ap.parse_args())
filename = args["audio"]
#file = wave.open(args["audio"], "r")

stripFile, fileType = func.noFileType(filename)

print(fileType)
if (fileType == 'mp4'):
    print('yes')
    if not (path.exists(stripFile +".wav")):
        print("Extracting audio from video")
        clip = mp.VideoFileClip(filename)
        clip.audio.write_audiofile(stripFile + '.wav')

sRate, data = read(stripFile + '.wav')
if (data.shape[1] > 1):
    data = data[:, 0] #звук двухканальный, оставляем только один
time = np.arange(0, len(data)/sRate, 1/sRate)

print("Normalizing audio")
data = func.signalNormalization(data, 0.95, 10000)

frame_size = sRate//8
print("frame size: ", frame_size)
print("len(data): ", len(data))
steps = len(data)//frame_size
timeStep = frame_size/sRate
timeArray = np.arange(0, timeStep*steps, timeStep)
freqArray = np.zeros(math.ceil(frame_size/2))

dt = 1/sRate
freq_vector = np.fft.rfftfreq(frame_size, d=dt)

#fft_plot(data, sRate)
steps = 500
spectrogram = np.zeros((steps, math.ceil((frame_size+1)/2)))
plotting = True
print(steps, math.ceil(frame_size/2))
print("Applying fast fourier transform")
for i in range(steps):
    print(str(i) + "/" + str(steps), end="\r")
    #fft_pl(data[i*sRate:(i+1)*sRate], sRate)
    signal = data[i*frame_size:(i+1)*frame_size]
    #print("len: ", len(signal))
    #print(len(signal))
    if(i == steps - 1):
        freqArray = freq_vector
    X = np.fft.rfft(signal)
    spectrogram[i] = X
        #plt.show()

    diff_X = np.concatenate( [ np.array([0]), np.abs(X)[1:]-np.abs(X)[:-1] ] )

    log_X = np.log(np.abs(X))
    cepstrum = np.fft.rfft(X)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(X.size, df)
    for i in range(len(cepstrum)):
        if (i > 200):
            cepstrum[i] = 0
    inverted = np.fft.irfft(cepstrum)

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

xlim = func.hzIndex(freq_vector, 2000)
print("xlim = ", xlim)
X = freq_vector[1:xlim]
k = 0
plotStart = 0
plotEnd = 100
for i in np.arange(plotStart, plotEnd):
    plt.figure(figsize=(20,10))
    Y = np.abs(spectrogram[i][1:xlim])
    Ymax = np.amax(Y)
    maxIndex = np.argmax(Y)
    plt.subplot(311)
    plt.plot(X, Y, "-", color=(0.9, 0.9, 0.9))
    lines = np.array([X[maxIndex], X[maxIndex]])
    colors = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)]
    plt.plot(X[maxIndex], Ymax, "g.")
    #plt.legend()
    #plt.show()
    plt.ylim(0,10000000)
    plt.xlim(X[0], X[-1])
    stepX2, stepTrueX2, stepMax2 = func.steppingMax(X, Y, 4, 1)
    #plt.plot(stepX2, stepMax2, "-", color=(0, 1.0, 1.0))
    stepX, stepTrueX, stepMax = func.steppingMax(X, Y, 200, 100)
    plt.plot(stepX, stepMax, ".", color=(1.0, 0.9, 0.9))
    plt.plot(stepTrueX, stepMax, '.', color=(0.0, 0.5, 0.0))

    xmax, maxs = func.catchingMax(X, stepTrueX, 25, 1)
    freq = xmax[np.argmax(maxs)]
    for j in range(1, 9):
        plt.plot([freq*j-25, freq*j, freq*j+25], [0,0,0], "r.-")
    #stepValues = func.removeRepeat(stepMax)
    #print(stepValues)
    #picks = len(stepValues)
    #good = func.goodness(picks, 5, 6)
    #print(good)
    #if good:
    #    clr = (0, 1.0, 0)
    #else:
    #    clr = (1.0, 0, 0)
    #plt.title(str(picks) + " picks", color=clr)

    plt.subplot(312)
    plt.plot(xmax, maxs)

    #areaX, areaSum = resonance(X, Y, 6, 10)
    #plt.plot(areaX, areaSum)
    #33plt.ylim(0,40000000)

    plt.subplot(313)
    audRange = 4
    chunkStart = max(0, frame_size*(i-audRange))
    chunkEnd = min(len(data), frame_size*(i+1+audRange))
    frameStart = max(0, frame_size*i)
    frameEnd = min(len(data), frame_size*(i+1))
    #print("frame_size, chunkStart, chunkEnd = ", frame_size, chunkStart, chunkEnd)
    dataX = np.arange(chunkStart,chunkEnd)
    dataChunk = data[chunkStart:chunkEnd]
    frameX = np.arange(frameStart, frameEnd)
    frameY = data[frameStart:frameEnd]
    #print(len(dataX))
    #print(len(dataChunk))
    plt.plot([dataX[0], dataX[-1]], [0, 0], "-", color=(0.2, 0.2, 0.2))
    plt.plot(dataX, dataChunk)
    bendingX, bendingY = bendingMax(frameX, frameY, 500, 100)
    minSignal = np.min(bendingY)
    maxSpectre = np.max(Y)
    #print("minSignal = ", minSignal)
    #print("maxSpectre = ", maxSpectre)
    if (minSignal > 1000 and maxSpectre > 2000000):
        print("freq = ", freq)
        plt.title("good", color=(0.0, 1.0, 0.0))
    else:
        plt.title("bad", color=(1.0, 0.0, 0.0))
    plt.plot(bendingX, bendingY)
    plt.plot([frame_size*i, frame_size*(i+1)], [0,0])
    #plt.ylim((-30000, 30000))
    plt.show()


#plt.plot(data)
#plt.show()
