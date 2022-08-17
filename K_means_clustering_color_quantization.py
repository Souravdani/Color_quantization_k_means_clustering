# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:15:04 2022
K_means_clustering_color_quantization
@author: Soura
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Converting .JPG/ .PNG etc into a numpy array
img_as_array= mpimg.imread("C:\\Users\\Soura\\Downloads\\DATA\\palm_trees.JPG")
img_as_array.shape  # (1401, 934, 3) ---> (H,M,C)

plt.figure(dpi= 300)
plt.imshow(img_as_array) # plt.imshow needs numpy array to show the image

## (H,M,C)----> 2D (H*M,C)
(h,w,c)= img_as_array.shape

## Reshaping
img_as_array2D= img_as_array.reshape(h*w,c)
img_as_array2D
len(img_as_array2D.shape)


from sklearn.cluster import KMeans
model= KMeans(n_clusters= 6)

labels= model.fit_predict(img_as_array2D)
labels

model.cluster_centers_
rgb_codes= model.cluster_centers_.round(0).astype(int)
rgb_codes

rgb_codes[labels]
quantized_image= np.reshape(rgb_codes[labels],(h,w,c))
len(quantized_image.shape) ## Back into 3-D array

## Now displaying our quantized image
plt.figure(dpi=300)
plt.imshow(quantized_image)
















