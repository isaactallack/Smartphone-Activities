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
x_gyro_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_gyro_x_train.csv", delimiter = ",")
y_gyro_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_gyro_y_train.csv", delimiter = ",")
z_gyro_v = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_gyro_z_train.csv", delimiter = ",")
x_gyro_fft = np.genfromtxt("D:\\Desktop\\x_gyro_fourier.csv", delimiter = ',')
y_gyro_fft = np.genfromtxt("D:\\Desktop\\y_gyro_fourier.csv", delimiter = ',')
z_gyro_fft = np.genfromtxt("D:\\Desktop\\z_gyro_fourier.csv", delimiter = ',')
lbls = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

length = len(x_fft)
x = sy.linspace(0.02, length*0.02, num=length)
f = syfp.fftfreq(length, np.mean(np.diff(x)))

std_vals_array = []

def comp_accel_ftrs(t_values, fft_values):
    MSMMM = [np.mean(t_values), np.std(t_values), np.median(t_values), np.amax(t_values), np.amin(t_values)]
    ASoS = [(sum(val*val for val in t_values))/128]
    IQR = [np.subtract(*np.percentile(t_values, [75, 25]))]
    f_max_mag = [np.amax(fft_values)]
    f_max_freq = [abs(f[list(fft_values).index(f_max_mag)])]
    f_MSM = [np.mean(fft_values), np.std(fft_values), np.median(fft_values)]
    f_skew = [sy.stats.skew(fft_values)]
    f_kurtosis = [sy.stats.kurtosis(fft_values)]
    f_IQR = [np.subtract(*np.percentile(fft_values, [75, 25]))]
    Energy = [(sum(f_val*f_val for f_val in fft_values))/128]
    return t_values.tolist()+fft_values.tolist()+MSMMM+ASoS+IQR+f_max_mag+f_max_freq+f_skew+f_kurtosis+f_IQR+f_MSM+Energy
#t_values.tolist()+fft_values.tolist()+

def comp_gyro_ftrs(t_gyro_values, gyro_fft_values):
    MSMMM = [np.mean(t_gyro_values), np.std(t_gyro_values), np.median(t_gyro_values), np.amax(t_gyro_values), np.amin(t_gyro_values)]
    ASoS = [(sum(val*val for val in t_gyro_values))/128]
    IQR = [np.subtract(*np.percentile(t_gyro_values, [75, 25]))]
    f_max_mag = [np.amax(gyro_fft_values)]
    f_max_freq = [abs(f[list(gyro_fft_values).index(f_max_mag)])]
    f_MSM = [np.mean(gyro_fft_values), np.std(gyro_fft_values), np.median(gyro_fft_values)]
    f_skew = [sy.stats.skew(gyro_fft_values)]
    f_kurtosis = [sy.stats.kurtosis(gyro_fft_values)]
    f_IQR = [np.subtract(*np.percentile(gyro_fft_values, [75, 25]))]
    Energy = [(sum(f_val*f_val for f_val in gyro_fft_values))/128]
    #return MSMMM+ASoS+IQR+f_max_mag+f_max_freq+f_skew+f_kurtosis+f_IQR+f_MSM+Energy
    return f_max_mag+f_max_freq+f_skew+f_kurtosis+f_IQR+f_MSM+Energy
#t_gyro_values.tolist()+gyro_fft_values.tolist()

for i in range(0, len(x_v)):
    std_vals_array.append(comp_accel_ftrs(x_v[i], x_fft[i])
                          +comp_accel_ftrs(y_v[i], y_fft[i])
                          +comp_accel_ftrs(z_v[i], z_fft[i])
                          +comp_gyro_ftrs(x_gyro_v[i], x_gyro_fft[i])
                          +comp_gyro_ftrs(y_gyro_v[i], y_gyro_fft[i])
                          +comp_gyro_ftrs(z_gyro_v[i], z_gyro_fft[i]))

X_train, X_test, y_train, y_test = train_test_split(std_vals_array, lbls, test_size=0.20)

classifier = svm.SVC()
#classifier = KNeighborsClassifier(n_neighbors=15)  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

numpy_train = np.array(X_train)

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


    
    
    
