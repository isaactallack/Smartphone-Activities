import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import timeit
  

start = timeit.default_timer()
xfourier = np.genfromtxt("D:\\Desktop\\x_values_fourier.csv", delimiter = ',')
yfourier = np.genfromtxt("D:\\Desktop\\y_values_fourier.csv", delimiter = ',')
zfourier = np.genfromtxt("D:\\Desktop\\z_values_fourier.csv", delimiter = ',')
x_gyro_fourier = np.genfromtxt("D:\\Desktop\\x_gyro_fourier.csv", delimiter = ',')
y_gyro_fourier = np.genfromtxt("D:\\Desktop\\y_gyro_fourier.csv", delimiter = ',')
z_gyro_fourier = np.genfromtxt("D:\\Desktop\\z_gyro_fourier.csv", delimiter = ',')

labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

length = len(xfourier[0])
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

mags = []
magn = []
n_bins = 128

spacing = int(length/n_bins)

for i in range(len(labels)):
        mags = []
        for j in range(0,n_bins):
                magx = xfourier[i][0+(spacing*j)]
                magy = yfourier[i][0+(spacing*j)]
                magz = zfourier[i][0+(spacing*j)]
                g_magx = x_gyro_fourier[i][0+(spacing*j)]
                g_magy = y_gyro_fourier[i][0+(spacing*j)]
                g_magz = z_gyro_fourier[i][0+(spacing*j)]
                mags.append(magx)
                mags.append(magy)
                mags.append(magz)
                mags.append(g_magx)
                mags.append(g_magy)
                mags.append(g_magz)
        magn.append(mags)

X_train, X_test, y_train, y_test = train_test_split(magn, labels, test_size=0.20)

classifier = svm.SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

numpy_train = np.array(X_train)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

stop = timeit.default_timer()
print('Time: ', stop - start)


