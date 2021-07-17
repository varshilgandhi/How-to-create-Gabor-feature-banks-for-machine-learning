# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:31:15 2021

@author: abc
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#let's it visualize using an image
img = cv2.imread('synthetic.jpg')

#Convert this image into gray level image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Convert image into single dimension
img2 = img.reshape(-1)

#Create dataframe 
df = pd.DataFrame()

#Add column name to our dataframe
df['Original Pixels'] = img2

#Create a range of values for theta, sigma, lamda, gamma

num = 1
for theta in range(2):
    theta = theta/4.* np.pi
    for sigma in (3,5):
        for lamda in np.arange(0, np.pi, np.pi/4.) :
            for gamma in (0.05, 0.5):
                gabor_label = 'Gabor' + str(num)
                kernel = cv2.getGaborKernel((5, 5), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img
                num += 1
                
                
print(df.head())
df.to_csv('Gabor_csv')
            
               



