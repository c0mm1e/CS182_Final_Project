from PIL import Image
from pylab import *
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(x, y):
    return sum([x[i] == y[i] for i in range(len(x))])/len(x)

# 数据处理
def dataProcessor(img_path):
    img = cv2.imread(img_path, 0)
    img = remove_black_edges(img)
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram = plt.hist(pixelSequence, numberBins, facecolor='black', histtype='bar')[0]
    frequency = histogram / h / w

    max_frequency = max(frequency)
    # the max_frequency of Nijigen will be bigger
    num_peakpoint = sum([frequency[i] > 5 * frequency[i - 1] and frequency[i] > 5 * frequency[i + 1] for i in range(1, 255)])
    # the num_peakpoint of Nijigen will be bigger

    return max_frequency, num_peakpoint

    # plt.xlabel("gray label")
    # plt.ylabel("number of pixels")
    # plt.axis([0, 255, 0, np.max(histogram)])
    # plt.show()

# 观测关联
def imagesProcessor(nijigen_directory, sanjigen_directory, num):
    nijigen_image_list = os.listdir(nijigen_directory)[:num]
    sanjigen_image_list = os.listdir(sanjigen_directory)[:num]
    max_frequency_list, num_peakpoint_list = [], []
    for image in nijigen_image_list:
        image_path = nijigen_directory + '/' + image
        max_frequency, num_peakpoint = dataProcessor(image_path)
        max_frequency_list.append(max_frequency)
        num_peakpoint_list.append(num_peakpoint)
    for image in sanjigen_image_list:
        image_path = sanjigen_directory + '/' + image
        max_frequency, num_peakpoint = dataProcessor(image_path)
        max_frequency_list.append(max_frequency)
        num_peakpoint_list.append(num_peakpoint)
    y = [1 for i in range(num)] + [0 for i in range(num)]
    x = [[max_frequency_list[i], num_peakpoint_list[i]] for i in range(2*num)]
    y = np.asarray(y)
    x = np.asarray(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max() 
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    plt.cla()
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  
    plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.show()


    # plt.cla()
    # plt.scatter(max_frequency_list[:num], num_peakpoint_list[:num])
    # plt.scatter(max_frequency_list[num:2*num], num_peakpoint_list[num:2*num])
    # plt.show()


#去黑边
def remove_black_edges(image):
    x = image.shape[0]
    y = image.shape[1]
    bottom = 0
    top = x - 1
    left = 0
    right = y - 1
    while sum([image[bottom,i] for i in range(y)]) == 0:
        bottom += 1
    while sum([image[top,i] for i in range(y)]) == 0:
        top -= 1
    x = image.shape[0]
    while sum([image[i,left] for i in range(x)]) == 0:
        left += 1
    while sum([image[i,right] for i in range(x)]) == 0:
        right -= 1        
    image2=image[bottom:top,left:right]
    return image2


# SVM
imagesProcessor(r"../Dataset/danbooru-images/0000", r"../Dataset/imagenet/train", 500)