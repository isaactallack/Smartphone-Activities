import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix  
  

xfourier = np.genfromtxt("D:\\Desktop\\x_values_fourier.csv", delimiter = ',')
labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

length = len(xfourier[0])
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

magn_freq = []
newlabels = []

# 1: Walking
# 2: Walking upstairs
# 3: Walking downstairs
# 4: Sitting
# 5: Standing
# 6: Laying
activity_1 = 1

for i in range(len(labels)):
    if labels[i] == activity_1:
        maxmag = np.amax(xfourier[i])
        maxfreq = abs(f[list(xfourier[i]).index(maxmag)])
        magn_freq.append([maxfreq, maxmag])
        newlabels.append(labels[i])

X_train, X_test, y_train, y_test = train_test_split(magn_freq, newlabels, test_size=0.20)

numpy_train = np.array(X_train)

plot_array_1 = []

for i in range(len(X_train)):
    if y_train[i] == activity_1:
        plot_array_1.append(numpy_train[i])
   
plot_array_1 = np.array(plot_array_1)

x = plot_array_1[:,0]
y = plot_array_1[:,1]

plt.hist2d(x,y,bins=17, range=[[0, 6], [0, 42]], vmin=0, vmax=250)

plt.colorbar(norm=mcolors.NoNorm)

plt.show()


