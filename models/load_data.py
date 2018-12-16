import cv2
import glob
from keras.utils import Sequence
import numpy as np
import random

def get_image_arr(path, width, height, img_norm = 'sub_mean', ordering = 'channels_first'):
    
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        if img_norm == 'sub_and_divide':
            # Gives image with values in [-1, 1]
            img /= 127.5 
            img -= 1
        elif img_norm == 'sub_mean':
            # Recommended preprocessing for VGG
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
        elif img_norm == 'divide':
            # Gives image with values in [0, 1]
            img = img / 255.

    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
    
    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def get_annotation_arr(path, width, height, n_classes):

    annotation = np.zeros((height, width, n_classes))
    
    try:
        img = cv2.imread(path)
        
        if len(img.shape) == 3:
            
            flattened = np.zeros((img.shape[0], img.shape[1]))
            for c in range(n_classes):
                mask = img == c
                mask = np.sum(mask, -1) >= 2
                flattened += mask * c
            img = flattened.astype('uint8')
        
        img = cv2.resize(img, (width, height))

        for c in range(n_classes):
            annotation[:,:,c] = (img == c).astype(int)

    except Exception as e:
        print(path, e)
        
    annotation = np.reshape(annotation, (width * height, n_classes))
    return annotation

def load_images(dir_images, width, height):
    
    assert dir_images[-1] == '/'
    path_images = glob.glob(dir_images + '*.png')
    path_images.sort()
    image_names = [path_image.split('/')[-1] for path_image in path_images]
    images = list(map(
        lambda path_img: get_image_arr(path_img, width, height),
        path_images
    ))
    return images, image_names

def load_annotations(dir_annotations, width, height, n_classes):
    
    assert dir_annotations[-1] == '/'
    
    path_annotations = glob.glob(dir_annotations + '*.png')
    path_annotations.sort()
    annotation_names = [path_annotation.split('/')[-1] for path_annotation in path_annotations]
    annotations = list(map(
        lambda path_ann: get_annotation_arr(path_ann, width, height, n_classes),
        path_annotations
    ))
    return annotations, annotation_names

class DataGenerator(Sequence):

    def __init__(self, dir_images, dir_annotations, batch_size, n_classes, 
        input_width, input_height, output_width, output_height):
        
        X, x_names = load_images(dir_images, input_height, input_height)
        y, y_names = load_annotations(dir_annotations, output_width, output_height, n_classes)
        
        assert len(X) == len(y)
        for x_name, y_name in zip(x_names, y_names):
            assert(x_name == y_name)
            
        paired = list(zip(X, y))
        random.shuffle(paired)
        X[:], y[:] = zip(*paired)

        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, batch_idx):
        batch_X = self.X[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        batch_y = self.y[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        return np.array(batch_X), np.array(batch_y)