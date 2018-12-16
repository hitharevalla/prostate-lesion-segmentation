import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, jaccard_similarity_score

from load_data import *
from model_definitions.segnet import *
from model_definitions.bayesian_segnet import *
from model_definitions.vgg_segnet import *
from model_definitions.vgg_bayesian_segnet import *
from model_definitions.unet import *
from model_definitions.unet_padded import *
from model_definitions.vgg_unet import *
from model_definitions.vgg_unet_padded import *
from utils import *

def my_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float(intersection) / union

def predict(dir_images, dir_annotations, dir_predictions, 
            n_classes, input_width, input_height, 
            model_name, epoch, loss, optimizer, metrics,
            path_load_weights):
    
    initialize_dir(dir_predictions) # Deletes files in dir_predictions
    
    model_architectures = {
        'segnet': segnet,
        'bayesian_segnet': bayesian_segnet,
        'vgg_segnet': vgg_segnet,
        'vgg_bayesian_segnet': vgg_bayesian_segnet,
        'unet': unet,
        'unet_padded': unet_padded,
        'vgg_unet': vgg_unet,
        'vgg_unet_padded': vgg_unet_padded,
    }
    
    model_func = model_architectures[model_name]
    m = model_func(n_classes, input_width, input_height)
    m.load_weights(path_load_weights + "." + str(epoch - 1))
    m.compile(loss = loss, optimizer = optimizer, metrics = metrics)

    output_height = m.output_height
    output_width = m.output_width

    path_images = glob.glob(dir_images + "*.png")
    path_images.sort()
    
    path_annotations = glob.glob(dir_annotations + "*.png")
    path_annotations.sort()
    
    X = [get_image_arr(path, input_width, input_height) for path in path_images]
    
    y_true = [get_annotation_arr(path, output_width, output_height, n_classes) for path in path_annotations]
    y_true = [true.reshape((output_width, output_height, n_classes)) for true in y_true]
    y_true = [true.argmax(axis = -1) for true in y_true]
    y_true = np.array(y_true)
    
    y_pred = [m.predict(np.array([image]))[0] for image in X]
    y_pred = [pred.reshape((output_width, output_height, n_classes)) for pred in y_pred]
    y_pred = [pred.argmax(axis = -1) for pred in y_pred]
    y_pred = np.array(y_pred)
    
    print(classification_report(y_true.flatten(), y_pred.flatten()))
    print('F1/dice (binary): ', f1_score(y_true.flatten(), y_pred.flatten(), pos_label = 1, average = 'binary'))
    print('Jaccard/IOU: ', my_iou(y_true.flatten(), y_pred.flatten()))

    for pred, path_image in zip(y_pred, path_images):
        
        pred_out = np.zeros((output_height, output_width))
        for c in range(n_classes):
            pred_out += ((pred == c) * c).astype('uint8')
        pred_out = cv2.resize(pred_out, (input_width, input_height))
        path_image_out = path_image.replace(dir_images, dir_predictions)
        cv2.imwrite(path_image_out, pred_out)