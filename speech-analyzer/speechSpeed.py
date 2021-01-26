import matplotlib.pyplot as plt
import numpy as np
import functions as func
import json

def getSpeechSpeed(filename, ifLim, limit):
    sRate, data = func.get_data(filename, ifLim, limit)
    frame_size = sRate//8
    steps = len(data)//frame_size

    slice = np.arange(len(data))
    step=500
    bX, bY = func.bendingMax(slice, np.abs(data), r=step, step=step)
    threshold = np.percentile(np.abs(data), 99)*0.25
    speech = np.array([int(y > threshold) for y in bY])

    interval = 15
    stepsInSec = sRate/step
    fullIntervals = len(data)//(sRate*interval)
    silence = np.zeros(fullIntervals+1)
    words = np.zeros(fullIntervals+1)
    for i in range(len(speech)):
        #print(bX[i], (sRate*interval), bX[i]//(sRate*interval))
        silence[int(bX[i])//(sRate*interval)] += int(speech[i])
        if(speech[i] != speech[max(0, i-1)]):
            words[int(bX[i])//(sRate*interval)] += 1

    silence /= stepsInSec
    outputY = words/silence
    outputX = np.arange(fullIntervals+1)*interval
    
    output = {
        'x': outputX.tolist(),
        'y': outputY.tolist()
    }
    return output

if __name__ == '__main__':
    filename, limit, ifLim = func.get_args()
    output = getSpeechSpeed(filename, ifLim, limit)

    plt.plot(output['x'], output['y'])
    plt.title("Words per second")
    plt.show()
