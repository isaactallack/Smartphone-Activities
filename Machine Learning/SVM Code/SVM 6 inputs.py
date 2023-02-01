import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
  

xfourier = np.genfromtxt("D:\\Desktop\\x_values_fourier.csv", delimiter = ',')
yfourier = np.genfromtxt("D:\\Desktop\\y_values_fourier.csv", delimiter = ',')
zfourier = np.genfromtxt("D:\\Desktop\\z_values_fourier.csv", delimiter = ',')
labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

length = len(xfourier[0])
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

magn_freq = []

for i in range(len(labels)):
        maxmagx = np.amax(xfourier[i])
        maxmagy = np.amax(yfourier[i])
        maxmagz = np.amax(zfourier[i])
        maxfreqx = abs(f[list(xfourier[i]).index(maxmagx)])
        maxfreqy = abs(f[list(yfourier[i]).index(maxmagy)])
        maxfreqz = abs(f[list(zfourier[i]).index(maxmagz)])
        magn_freq.append([maxfreqx, maxmagx, maxfreqy, maxmagy, maxfreqx, maxmagz])

X_train, X_test, y_train, y_test = train_test_split(magn_freq, labels, test_size=0.20)

classifier = svm.SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

numpy_train = np.array(X_train)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  


