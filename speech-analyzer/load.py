import matplotlib.pyplot as plt
import numpy as np

a = np.load('formants.npy')
goodness, formantFreq, formantValue = a

x_freq = np.array([i for i in range(len(goodness)) if goodness[i]])
y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if goodness[i]])
bad_x_freq = np.array([i for i in range(len(goodness)) if not goodness[i]])
bad_y_freq = np.array([formantFreq[i] for i in range(len(goodness)) if not goodness[i]])
plt.plot(np.arange(len(formantFreq))/8, formantFreq, '-', color=(0.8, 0.8, 0.8))
plt.plot(bad_x_freq/8, bad_y_freq, '.', color=(1.0, 0.6, 0.6))
plt.plot(x_freq/8, y_freq, '.')
plt.xticks(np.arange(len(formantFreq)//480)*60)
plt.grid()
plt.show()
