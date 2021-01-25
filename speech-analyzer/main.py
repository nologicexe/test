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



#-------Parsing arguments--------

ap = argparse.ArgumentParser()
ap.add_argument("audio", type=str, default="", help="path to input video file")
ap.add_argument("-l", "--limit", type=int, default=None, help="frame processing limit")

args = vars(ap.parse_args())
filename = args["audio"]
limit = args["limit"]
ifLim = (limit is not None)



#-------Extracting single channel audio wav file-------

stripFile, fileType = func.noFileType(filename)
if (fileType == 'mp4'):
    if not (path.exists(stripFile +".wav")):
        print("Extracting audio from video")
        clip = mp.VideoFileClip(filename)
        clip.audio.write_audiofile(stripFile + '.wav')

sRate, data = read(stripFile + '.wav')
if (len(data.shape) > 1):
    data = data[:, 0] #звук двухканальный, оставляем только один
time = np.arange(0, len(data)/sRate, 1/sRate)

print("Normalizing audio")
data = func.signalNormalization(data, 0.95, 10000)



#-------Processing--------

frame_size = sRate//8
steps = len(data)//frame_size
if (ifLim and limit < steps):
    steps = limit
timeStep = frame_size/sRate
timeArray = np.arange(0, timeStep*steps, timeStep)
freqArray = np.zeros(math.ceil(frame_size/2))
dt = 1/sRate
freq_vector = np.fft.rfftfreq(frame_size, d=dt)
xlim = func.hzIndex(freq_vector, 2000)
X = freq_vector[:xlim]
spectrogram = np.zeros((steps, xlim))
print("frame_size = ", frame_size)
print("len)freq) = ", len(freq_vector))
print("steps = ", steps)
print("Applying fast fourier transform")
for i in range(steps):
    #print(str(i) + "/" + str(steps), end="\r")
    signal = data[i*frame_size:(i+1)*frame_size]
    Y = np.fft.rfft(signal)
    Y = np.abs(Y[:xlim])
    stepX, stepTrueX, stepMax = func.steppingMax(X, Y, 200, 100)
    IDs, xmax, maxes = func.catchingMax(X, stepTrueX, 25, 1)
    freqID = IDs[np.argmax(maxes)]
    Xsignal = np.arange(i*frame_size, (i+1)*frame_size)
    bendingX, bendingY = func.bendingMax(Xsignal, signal, 500, 100)
    minSignal = np.min(bendingY)
    maxSpectre = np.max(Y)
    if(minSignal > 1000 and maxSpectre > 2000000):
        Y[freqID] = -10000000
        print(freqID, i)
    else:
        Y = np.ones(xlim)*-5000000
    spectrogram[i] = Y

print(spectrogram[140])

#------plotting spectrogram-----------

fig, ax = plt.subplots(figsize=(20, 10))
t = np.arange(spectrogram.shape[0])/8
w = freq_vector[:xlim]
xv, yv = np.meshgrid(t, w)
z = spectrogram[:,:xlim].T
c = ax.pcolormesh(xv, yv, z, cmap="RdBu", vmin=-10000000, vmax=10000000)
ax.axis([t.min(), t.max(), w.min(), w.max()])
fig.colorbar(c, ax=ax)
plt.show()

#-------Postprocessing--------------


print("xlim = ", xlim)
X = freq_vector[:xlim]
k = 0
plotStart = 0
plotEnd = 100
for i in np.arange(plotStart, plotEnd):
    Y = np.abs(spectrogram[i][:xlim])
    Ymax = np.amax(Y)
    maxIndex = np.argmax(Y)
    lines = np.array([X[maxIndex], X[maxIndex]])
    stepX, stepTrueX, stepMax = func.steppingMax(X, Y, 200, 100)
    IDs, xmax, maxs = func.catchingMax(X, stepTrueX, 25, 1)
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
    bendingX, bendingY = func.bendingMax(frameX, frameY, 500, 100)
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


#plt.plot(data)
#plt.show()
