import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import csv
import math

function = []

frequency = 2
frequency2 = 5

length = 256
x = sy.linspace(0.00, length*0.02, num=length)
for p in x:
    trigvalue = math.cos(2*math.pi*frequency*p)+math.cos(2*math.pi*frequency2*p)
    function.append(trigvalue)

yf = syfp.fft(function)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

plt.subplot(2,1,1)
plt.plot(x, function, 'r-')

plt.subplot(2,1,2)
plt.plot(abs(f), abs(yf), 'b-')

plt.show()
