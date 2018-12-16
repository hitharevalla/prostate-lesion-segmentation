import argparse
import numpy as np
from skimage.transform import resize
from utils import *

def add_image_num_to_name(images):
    images_renamed = list_map(
        lambda img, num: img.update_name('_'.join([img.name, str(num)])),
        images, range(len(images))
    )
    return images_renamed

def reflect_image(image):
    image_lr = Image(image.name, np.fliplr(image.pixel_array))
    return [image, image_lr]

def rotate_image(image):
    image_90 = Image(
        image.name, 
        np.rot90(image.pixel_array, axes = (0, 1))
    )
    image_180 = Image(
        image.name, 
        np.rot90(image_90.pixel_array, axes = (0, 1))
    )
    image_270 = Image(
        image.name, 
        np.rot90(image_180.pixel_array, axes = (0, 1))
    )
    return [image, image_90, image_180, image_270]

def augment_image(image):
    images_reflected = reflect_image(image)
    images_rotated = list_map(
        lambda img: rotate_image(img), 
        images_reflected
    )
    return flatten(images_rotated)

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-out', type = str, required = True)
    args = parser.parse_args()
    return args
        
def main():
    paths = handle_args()
    images = load_images(paths.images)
    images_augmented = list_map(
        lambda img: augment_image(img),
        images
    )
    images_renamed = list_map(
        lambda imgs: add_image_num_to_name(imgs),
        images_augmented
    )
    write_images(paths.out, flatten(images_renamed))
    
if __name__ == '__main__':
    main()