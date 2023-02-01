import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import random

xfourier = np.genfromtxt("D:\\Desktop\\x_values_fourier.csv", delimiter = ',')
xvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_x_train.csv", delimiter = ",")
yfourier = np.genfromtxt("D:\\Desktop\\y_values_fourier.csv", delimiter = ',')
yvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_y_train.csv", delimiter = ",")
zfourier = np.genfromtxt("D:\\Desktop\\z_values_fourier.csv", delimiter = ',')
zvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_z_train.csv", delimiter = ",")
labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

labels_dic = { # create a dictionary with the label numbers and corresponding activities
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
    }

graphed = []

length = len(xvalues[0])
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

while len(graphed) != 6:
    randnum = random.randint(0, len(xvalues))
    if labels[randnum] not in graphed:
        graphed.append(labels[randnum])
        plt.figure(len(graphed))
        plt.suptitle(labels_dic.get(labels[randnum]))
        
        plt.subplot(2,3,1)
        plt.plot(x, xvalues[randnum], 'b-')
        plt.title('x-direction')

        plt.subplot(2,3,2)
        plt.plot(x, yvalues[randnum], 'b-')
        plt.title('y-direction')

        plt.subplot(2,3,3)
        plt.plot(x, zvalues[randnum], 'b-')
        plt.title('z-direction')

        plt.subplot(2,3,4)
        plt.plot(abs(f), xfourier[randnum], 'r-')

        plt.subplot(2,3,5)
        plt.plot(abs(f), yfourier[randnum], 'r-')

        plt.subplot(2,3,6)
        plt.plot(abs(f), zfourier[randnum], 'r-')

plt.show()
