import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import random
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

activity_1 = 1
activity_2 = 4

for i in range(len(labels)):
    if ((labels[i] == activity_1) or (labels[i] == activity_2)):
        maxmag = np.amax(xfourier[i])
        maxfreq = abs(f[list(xfourier[i]).index(maxmag)])
        magn_freq.append([maxfreq, maxmag])
        newlabels.append(labels[i])

X_train, X_test, y_train, y_test = train_test_split(magn_freq, newlabels, test_size=0.20)

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

numpy_train = np.array(X_train)

plot_array_1 = []
plot_array_2 = []
for i in range(len(X_train)):
    if y_train[i] == activity_1:
        plot_array_1.append(numpy_train[i])
    if y_train[i] == activity_2:
        plot_array_2.append(numpy_train[i])
   
plot_array_1 = np.array(plot_array_1)
plot_array_2 = np.array(plot_array_2)
plt.scatter(plot_array_1[:,0], plot_array_1[:,1], marker = '.', s = 40, color = 'r')
plt.scatter(plot_array_2[:,0], plot_array_2[:,1], marker = '.', s = 40, color = 'b')

wrong_class = []
for i in range(len(y_test)):
    if (y_test[i]-y_pred[i]) != 0:
        wrong_class.append(X_test[i])

if len(wrong_class) != 0:
    wrong_class = np.array(wrong_class)
    plt.scatter(wrong_class[:,0], wrong_class[:,1], marker = 'x', s=30, color = 'g')

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

plt.show()


