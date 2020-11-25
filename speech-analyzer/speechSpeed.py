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

filename, limit, ifLim = func.get_args()
sRate, data = func.get_data(filename, ifLim, limit)
frame_size = sRate//8
steps = len(data)//frame_size
#spectrogram, steps, X, timeArray = func.get_spectrogram(sRate, data, frame_size, steps, ifLim, limit)
#formantID, formantFreq, formantValue = func.get_formants(spectrogram, X, steps)
#goodness, goodSum, signalMins, SpectrumMaxes = func.get_good(data, steps, frame_size, spectrogram)
#func.plot_bend(data, steps, frame_size, signalMins, SpectrumMaxes)
#func.plot_freq_response(formantFreq, formantValue, goodness)
#func.plot_spectrogram(timeArray, X, spectrogram, formantID)
#func.postprocessing(spectrogram, X, data, frame_size)

#for i in range(steps):
#slice = np.arange(i*frame_size,(i+1)*frame_size)
slice = np.arange(len(data))
step=500
bX, bY = func.bendingMax(slice, np.abs(data), r=step, step=step)
threshold = np.percentile(np.abs(data), 99)*0.25
speech = np.array([int(y > threshold) for y in bY])

interval = 60
fullIntervals = len(data)//(sRate*interval)
print(fullIntervals)
silence = np.zeros(fullIntervals+1)
for i in range(len(speech)):
    #print(bX[i], (sRate*interval), bX[i]//(sRate*interval))
    silence[int(bX[i])//(sRate*interval)] += int(speech[i])
silence /= (interval*sRate/step)
for s in range(len(silence)-1):
    print(s, silence[s])

plt.plot(slice/sRate, np.abs(data))
plt.plot(bX/sRate, bY)
plt.plot(bX[[0, -1]]/sRate, [threshold]*2)
plt.plot(bX/sRate, speech*max(data), color=(1.0, 0.0, 0.0, 0.5))
plt.show()

#график пауз, численное соотношение паузы к непаузе в интервал временной (минута)
