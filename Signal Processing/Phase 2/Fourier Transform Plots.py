import numpy as np
import scipy as sy
import scipy.fftpack as syfp
import matplotlib.pyplot as plt

values = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\inertial_signals\\body_acc_x_train.csv", delimiter = ",")
labels = np.genfromtxt("D:\\Desktop\\UCI HAR Dataset\\train\\y_train.txt", delimiter = "\n")

labels_dic = {
    1: "Walking",
    2: "Walking upstairs",
    3: "Walking downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
    }

def fourier_transform(num): # 1: walking, 2: walking upstairs, 3: walking downstairs
                            # 4: sitting, 5: standing, 6: laying
    array = []
                            
    for i in range(len(labels)-1):
        if labels[i] == num:
            array.append(values[i])

    array = np.asarray(array)
    array = np.ndarray.flatten(array)

    length = len(array)
    x = sy.linspace(0.02, length*0.02, num=length)

    yf = syfp.fft(array)
    f = syfp.fftfreq(length, np.mean(np.diff(x)))
    yf = syfp.fftshift(yf)

    if num <=3:
        plt.figure(1)
        plt.subplot(2, 3, num) #Create signal magnitude by time plot
        plt.plot(x[::200], array[::200], 'b-',) #Plot values
        plt.title(labels_dic.get(num))
    
        plt.subplot(2, 3, num+3) #Create magnitude by frequency plot
        plt.plot(abs(f), abs(yf), 'r-') #Plot values
        plt.suptitle('X-direction', fontsize=16)

    if num >= 4:
        plt.figure(2)
        plt.subplot(2, 3, num-3) #Create signal magnitude by time plot
        plt.plot(x[::200], array[::200], 'b-',) #Plot values
        plt.title(labels_dic.get(num))
    
        plt.subplot(2, 3, num) #Create magnitude by frequency plot
        plt.plot(abs(f), abs(yf), 'r-') #Plot values
        plt.suptitle('X-direction', fontsize=16)

for i in range(1,7):
    print(i)
    fourier_transform(i)

plt.show()
