import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt

array = np.loadtxt("D:\\Desktop\\accelxyz.csv", delimiter=',')
column_num = 2  #0 means x-axis, 1 means y-axis and 2 means z-axis

length = len(array[:,column_num]) #Number of data points
x = sy.linspace(0.005, length*0.008, num=length) #Return evenly spaced numbers as x-axis values

yf = syfp.fft(array[:,column_num]) #Discrete Fourier transform of array  
f = syfp.fftfreq(length, np.mean(np.diff(x))) #Return the Discrete Fourier Transform sample frequencies

plt.subplot(211) #Create signal magnitude by time plot
plt.plot(x, array[:,column_num]) #Plot values
plt.subplot(212) #Create magnitude by frequency plot
plt.plot(abs(f), abs(yf)) #Plot values
plt.show()
