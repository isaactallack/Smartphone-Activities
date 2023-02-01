import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import csv

xvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_x_train.csv", delimiter = ",")
yvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_y_train.csv", delimiter = ",")
zvalues = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_z_train.csv", delimiter = ",")
labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

store = []

def fourier_transform(array, activity):

    length = len(array)
    x = sy.linspace(0.02, length*0.02, num=length)

    yf = syfp.fft(array)
    f = syfp.fftfreq(length, np.mean(np.diff(x)))
    #yf = syfp.fftshift(yf)

    return f, yf, x

for i in range(0, len(xvalues)):
    f, yf, x = fourier_transform(xvalues[i], labels[i])
    store.append(abs(yf))
    print(i)

with open('x_values_fourier.csv', 'w') as csvFile:
    writer = csv.writer(csvFile, lineterminator = '\n')
    writer.writerows(store)
