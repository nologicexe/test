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
# print("len(data) = ", len(data))
# print("sRate = ", sRate)
# print("len(data)/sRate = ", len(data)/sRate)
# print("frame_size = ", frame_size)
# print("steps = ", steps)
spectrogram, X, timeArray = func.get_spectrogram(sRate, data, frame_size, steps)
#print("len(spectrogram) = ", len(spectrogram))
goodness, goodSum, signalMins, SpectrumMaxes = func.get_good(data, steps, frame_size, spectrogram)
formantID, formantFreq, formantValue = func.get_formants(spectrogram, X, steps)
save_array = np.array([goodness, formantFreq, formantValue])
np.save("formants.npy", save_array)
# x_freq = np.array([i for i in range(len(goodness)) if goodness[i]])
# y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if goodness[i]])
# bad_x_freq = np.array([i for i in range(len(goodness)) if not goodness[i]])
# bad_y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if not goodness[i]])
# plt.plot(np.arange(len(formantFreq))/8, formantFreq, '-', color=(0.8, 0.8, 0.8))
# plt.plot(x_freq/8, y_freq, '.')
# plt.plot(bad_x_freq/8, bad_y_freq, 'r.')
# plt.show()

# plt.plot(goodness)
# plt.show()
# func.plot_bend(data, steps, frame_size, signalMins, SpectrumMaxes)
# func.plot_freq_response(formantFreq, formantValue, goodness)
# func.plot_spectrogram(timeArray, X, spectrogram, formantID)
# func.postprocessing(spectrogram, X, data, frame_size)
