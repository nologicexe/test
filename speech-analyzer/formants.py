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
import json

def getFormants(filename, ifLim, limit, savefile=''):
    sRate, data = func.get_data(filename, ifLim, limit)
    frame_size = sRate//8
    steps = len(data)//frame_size

    spectrogram, X, timeArray = func.get_spectrogram(sRate, data, frame_size, steps)
    goodness, goodSum, signalMins, SpectrumMaxes = func.get_good(data, steps, frame_size, spectrogram)
    formantID, formantFreq, formantValue = func.get_formants(spectrogram, X, steps)

    output = {
        "goodness": goodness.tolist(),
        "formantFreq": formantFreq.tolist(),
        "formantValue": formantValue.tolist()
    }
    if (savefile != ''):
        with open(savefile, 'w') as json_file:
            json.dump(output, json_file)

    return output

if __name__ == "__main__":
    filename, limit, ifLim = func.get_args()
    output = getFormants(filename, ifLim, limit, savefile='test.json')
    goodness = output["goodness"]
    formantFreq = output["formantFreq"]
    formantValue = output["formantValue"]

    x_freq = np.array([i for i in range(len(goodness)) if goodness[i]])
    y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if goodness[i]])
    bad_x_freq = np.array([i for i in range(len(goodness)) if not goodness[i]])
    bad_y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if not goodness[i]])
    plt.plot(np.arange(len(formantFreq))/8, formantFreq, '-', color=(0.8, 0.8, 0.8))
    plt.plot(x_freq/8, y_freq, '.')
    #plt.plot(bad_x_freq/8, bad_y_freq, 'r.')
    plt.show()
