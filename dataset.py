#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:06:01 2017

@author: user
"""

import numpy as np
import os
import glob
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import filters
from skimage import io

def prepare_dataset(train_path, classes):
    labels = []
    images = []
    

    for className in classes:   
        index = classes.index(className)
        print('Reading {} files (Index: {})'.format(className, index))
        path = os.path.join(train_path, className, '*g')
        files = glob.glob(path)
        for fl in files:
            t_path = os.path.abspath(fl)
            labels.append(index)
            
            images.append(t_path)
      
    return np.array(images), np.array(labels)

def load_images(train_path, label, resize_img_height, resize_img_width, crop_h, crop_w):
    labels = []
    images = []
    

    for i in range(len(train_path)):   
        t_path = os.path.abspath(train_path[i])
        labels.append(label[i])
        #load image
        # first crop, then resize
        img_raw = rgb2gray(io.imread(t_path).astype(np.float) / 255)
        img_shape = np.array(img_raw).shape
        img_raw_cropped = img_raw[int(img_shape[0]*crop_h):int(img_shape[0]*(1-crop_h)),int(img_shape[1]*crop_w):int(img_shape[1]*(1-crop_w))]
        img = resize(img_raw_cropped,(resize_img_height, resize_img_width))
        #####val = filters.threshold_otsu(img)
        ####final = img < val
        #final = np.array(final)
        images.append(img)
        i+=1
        
    return np.array(images), np.array(labels)
    