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

#load and calculate formants
filename, limit, ifLim = func.get_args()
sRate, data = func.get_data(filename, ifLim, limit)
frame_size = sRate//8
steps = len(data)//frame_size

#plot audio
# for i in range(steps):
#     plt.plot(data[i*frame_size:(i+1)*frame_size])
#     plt.show()

spectrogram, X, timeArray = func.get_spectrogram(sRate, data, frame_size, steps)
goodness, goodSum, signalMins, SpectrumMaxes = func.get_good(data, steps, frame_size, spectrogram)
formantID, formantFreq, formantValue = func.get_formants(spectrogram, X, steps)

#save formant data to formants.npy
save_array = np.array([goodness, formantFreq, formantValue])
np.save("formants.npy", save_array)

#plot formant frequency
x_freq = np.array([i for i in range(len(goodness)) if goodness[i]])
y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if goodness[i]])
bad_x_freq = np.array([i for i in range(len(goodness)) if not goodness[i]])
bad_y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if not goodness[i]])
plt.plot(np.arange(len(formantFreq))/8, formantFreq, '-', color=(0.8, 0.8, 0.8))
plt.plot(x_freq/8, y_freq, '.')
#plt.plot(bad_x_freq/8, bad_y_freq, 'r.')
plt.show()

#plot spectrogram for every chunk
for i in range(len(spectrogram)):
    plt.plot(X, spectrogram[i])
    plt.title(str(i))
    plt.ylim(0, 1e7)
    plt.plot(formantFreq[i], formantValue[i], 'o', color=(1.0, 0.0, 0.0))
    plt.show()

# plt.plot(goodness)
# plt.show()
# func.plot_bend(data, steps, frame_size, signalMins, SpectrumMaxes)
# func.plot_freq_response(formantFreq, formantValue, goodness)
# func.plot_spectrogram(timeArray, X, spectrogram, formantID)
# func.postprocessing(spectrogram, X, data, frame_size)
#asdasdas
