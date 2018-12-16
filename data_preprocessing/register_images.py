import argparse
from itertools import groupby
from itertools import product
import matplotlib.pyplot as plt
from skimage import measure
from skimage import transform as tf
from utils import *

def get_transform_points(image):
    points = []
    min_row, min_col, max_row, max_col = measure.regionprops(image.pixel_array)[0].bbox
    points.append((min_row, min_col))
    points.append((max_row, max_col))
    points.append((min_row, max_col))
    points.append((max_row, min_col))
    points.append(measure.regionprops(image.pixel_array)[0].centroid)
    return np.array(points)

def register_modality(anchor, images):
    moving_image = list_filter(lambda img: img.typeof == 'ROI', images)[0]
    
    anchor_points = get_transform_points(anchor)
    moving_points = get_transform_points(moving_image)

    transform = tf.SimilarityTransform()
    transform.estimate(moving_points, anchor_points)
    
    images_registered = list_map(
        lambda img: img.update_pixel_array(tf.warp(img.pixel_array, transform)), 
        images
    )
    images_registered_standardized = list_map(
        lambda img: standardize_image(img),
        images_registered
    )
    
    return images_registered
        
def register_patient(images): 
    anchor = list_filter(
        lambda img: img.modality == 'T2' and img.typeof == 'ROI',
        images
    )[0]
    modality_dict = split_by_modality(images)
    images_registered = []
    for modality_images in modality_dict.values():
        images_registered.append(register_modality(anchor, modality_images))
    
    return flatten(images_registered)


def register_images(images):
    patient_dict = split_by_patient(images)
    registered_patients = list_map(
        lambda patient_images: register_patient(patient_images),
        patient_dict.values()
    )
    return flatten(registered_patients)

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-out', type = str, required = True)
    args = parser.parse_args()
    return args

def main():
    paths = handle_args()
    images = load_images(paths.images)
    images = register_images(images)
    write_images(paths.out, images)
    
if __name__ == '__main__':
    main()
