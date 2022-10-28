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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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

# 数据处理
def dataProcessor(img_path):
    img = cv2.imread(img_path, 0)
    img = remove_black_edges(img)
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram = plt.hist(pixelSequence, numberBins, facecolor='black', histtype='bar')[0]
    frequency = histogram / h / w
    return frequency

def KNN(nijigen_directory, sanjigen_directory, num):
    nijigen_image_list = os.listdir(nijigen_directory)[:num]
    sanjigen_image_list = os.listdir(sanjigen_directory)[:num]
    frequency_list = []
    for image in nijigen_image_list:
        image_path = nijigen_directory + '/' + image
        frequency_list.append(dataProcessor(image_path))  
    for image in sanjigen_image_list:
        image_path = sanjigen_directory + '/' + image
        frequency_list.append(dataProcessor(image_path)) 
    y = [1 for i in range(num)] + [0 for i in range(num)]
    y = np.asarray(y)
    x = np.asarray(frequency_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # 训练模型
    clf = KNeighborsClassifier(n_neighbors=17)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # 评价模型
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))

KNN(r"../Dataset/danbooru-images/0000", r"../Dataset/imagenet/train", 1000)
