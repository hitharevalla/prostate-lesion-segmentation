import cv2
import glob
import numpy as np
import os
import shutil

def initialize_dir(path_dir):
    
    if os.path.isdir(path_dir):
        shutil.rmtree(path_dir)
        
    os.makedirs(path_dir)

def calc_class_proportions(path_train_annotations, n_classes, n_channels):
    
    path_annotations = glob.glob(path_train_annotations + '*.png')
    class_counts = np.zeros(n_classes)
    for path_ann in path_annotations:
        for ch in range(n_channels):
            ann = cv2.imread(path_ann)[:,:,ch]
            ann_count = np.unique(ann, return_counts = True)
            class_counts = np.add(class_counts, ann_count[1])
    class_proportions = class_counts / np.sum(class_counts)
    return class_proportions

def calc_class_weights(path_train_annotations, n_classes, n_channels, scale = None):
    
    class_props = calc_class_proportions(path_train_annotations, n_classes, n_channels)
    if scale == 'log':
        weights = np.log(1 / class_props)
    else: 
        max_prop = np.max(class_props)
        weights = max_prop / class_props
    return weights