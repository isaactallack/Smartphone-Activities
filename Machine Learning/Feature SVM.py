import numpy as np
import scipy as sy
from scipy.special import entr
import scipy.fftpack as syfp
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm

x_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_x_train.csv", delimiter = ",")
y_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_y_train.csv", delimiter = ",")
z_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_z_train.csv", delimiter = ",")
x_fft = np.genfromtxt("D:\\Desktop\\x_values_fourier.csv", delimiter = ',')
y_fft = np.genfromtxt("D:\\Desktop\\y_values_fourier.csv", delimiter = ',')
z_fft = np.genfromtxt("D:\\Desktop\\z_values_fourier.csv", delimiter = ',')
lbls = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

length = len(x_fft)
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

std_vals_array = []

def comp_ftrs(t_values, fft_values):
    MSMMM = [np.mean(t_values), np.std(t_values), np.median(t_values), np.amax(t_values), np.amin(t_values)]
    ASoS = [(sum(val*val for val in t_values))/128]
    IQR = [np.subtract(*np.percentile(t_values, [75, 25]))]
    f_max_mag = [np.amax(fft_values)]
    f_max_freq = [abs(f[list(fft_values).index(f_max_mag)])]
    f_MSM = [np.mean(t_values), np.std(t_values), np.median(t_values)]
    f_skew = [sy.stats.skew(fft_values)]
    f_kurtosis = [sy.stats.kurtosis(fft_values)]
    f_IQR = [np.subtract(*np.percentile(fft_values, [75, 25]))]
    Energy = [(sum(f_val*f_val for f_val in fft_values))/128]
    return MSMMM+ASoS+IQR+f_max_mag+f_max_freq+f_skew+f_kurtosis+f_IQR+f_MSM+Energy


for i in range(0, len(x_v)):
    std_vals_array.append(comp_ftrs(x_v[i], x_fft[i])+comp_ftrs(y_v[i], y_fft[i])+comp_ftrs(z_v[i], z_fft[i]))

X_train, X_test, y_train, y_test = train_test_split(std_vals_array, lbls, test_size=0.20)

classifier1 = KNeighborsClassifier(n_neighbors=10)  
classifier1.fit(X_train, y_train)

classifier2 = svm.SVC()
classifier2.fit(X_train, y_train)

classifier3 = tree.DecisionTreeClassifier()
classifier3.fit(X_train, y_train)

y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)

numpy_train = np.array(X_train)

print(confusion_matrix(y_test, y_pred1))  
print(classification_report(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred2))  
print(classification_report(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred3))  
print(classification_report(y_test, y_pred3)) 


    
    
    
