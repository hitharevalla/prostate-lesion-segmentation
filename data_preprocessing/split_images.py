import argparse
import csv
from utils import *
from itertools import product
import os

def read_dict(path_csv):
    """
    Args:
        path_csv (string): Path to a csv file containing a dictionary
            written in the format
            key, value
            key, value
            ...
    Returns:
        dictionary (dict): A dict with string keys and string values
    """
    with open(path_csv, 'r') as csv_file:
        reader = csv.reader(csv_file)
        dictionary = dict(reader)
    return dictionary

def keep_Slice_ROI_pairs(images):
    """
    Args:
        images (list): A list of Image objects
    Returns:
        image_Slice_ROI_pairs (list): A list of Image objects with typeof
        'Slice' or 'ROI', where each Image has a pair in the list with the 
        same patient ID, slice number, and modality but opposite typeof
    """
    images_Slice_ROI = list_filter(
        lambda image: image.typeof == 'Slice' or image.typeof == 'ROI',
        images
    )
    images_Slice_ROI_pairs = list_filter(
        lambda image: bool(get_associated_images(image, images_Slice_ROI)),
        images_Slice_ROI
    )
    return images_Slice_ROI_pairs

def split_images_by_id(split_ids, images):
    """
    Args:
        split_ids (dict): A dict where the keys are the patient IDs as strings
            and the values are strings indicating which partition of the data 
            the patient's images belong to
        images (list): A list of image objects that need to be partitioned
            into train, validation, and test data sets
    Returns:
        images_splot (dict): A dict where the keys are one of 'train',
            'val', or 'test' and the values are lists of images in that
            parition of the data set
    """
    images.sort(key = lambda image: image.get_partition(split_ids))
    images_split = groupby(
        images,
        lambda image: image.get_partition(split_ids)
    )
    images_split = dict((part, list(imgs)) for part, imgs in images_split)
    return images_split

def name_without_typeof(img):
    name_parts = img.name.split('_')
    del name_parts[3]
    return '_'.join(name_parts)

def write_split_images(path, images_split):
    """
    Args:
        path (string): A path to the directory to create subdirectories
            for each split is
        images_split (dict): A dict where the keys are one of 'train',
            'val', or 'test' and the values are lists of images in that
            parition of the data set
    Returns:
        None
    """
    splits = images_split.keys()
    types = ['Slice', 'ROI']
    for split, typeof in product(splits, types):
        path_out = os.path.join(path, split, typeof)
        images_out = list_filter(
            lambda image: image.typeof == typeof,
            images_split[split]
        )
        write_images(path_out, images_out, name_func = name_without_typeof)

def handle_args():
    """
    Args:
        None
    Returns:
        args (object): An object with attributes corresponding to the
            required command line arguments 'images' and 'split_csv', where
            'images' is a path to a directory and 'split_csv' is a path to a
            csv file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-split_csv', type = str, required = True)
    args = parser.parse_args()
    return args

def main():

    paths = handle_args()

    ids_split = read_dict(paths.split_csv)
    images = load_images(paths.images)
    images_paired = keep_Slice_ROI_pairs(images)
    images_split = split_images_by_id(ids_split, images_paired)
    write_split_images(paths.images, images_split)

if __name__ == '__main__':
    main()
