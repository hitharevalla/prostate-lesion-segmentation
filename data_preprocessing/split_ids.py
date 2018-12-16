import argparse
import csv
from utils import *
from math import floor
import numpy as np
from random import seed
from random import shuffle

seed(1)

def partition_ids(unique_ids, train_prop, val_prop, test_prop):
    """
    Args:
        unique_ids (list): List of unique subject IDs in the data set as ints
        train_prop (float): Proportion of IDs to include in training data set
        val_prop (float): Proportion of IDs to include in validation data set
        test_prop (float): Proportion of IDs to include in testing data set
    Returns:
        split_ids (dictionary): A dictionary whose keys are the patient IDs as
            ints and whose values are the parition the patient belongs to, 
            i.e. one of 'train', 'validation', and 'test'
    """
    num_train = floor(train_prop * len(unique_ids))
    num_val = floor(val_prop * len(unique_ids))
    num_test = floor(test_prop * len(unique_ids))
    
    remainder = len(unique_ids) - num_train - num_val - num_test
    if remainder >= 1:
        num_train += 1
    if remainder >= 2:
        num_val += 1
    if remainder >= 3:
        num_test += 1

    shuffle(unique_ids)
    partition = ['train'] * num_train + ['val'] * num_val + ['test'] * num_test
    split_ids = dict(zip(unique_ids, partition))

    return split_ids

def write_dict_to_csv(dictionary, path_out_csv):
    """
    Args:
        dictionary (dictionary): Dictionary to be written to a csv in the form
            key, value
            key, value
            ...
        path_out_csv (string): The path to file to write the dictionary out 
            to in csv format
    """
    with open(path_out_csv, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])

def handle_args():
    """
    Args:
        None
    Returns:
        args (object): An object with attributes corresponsing to the 
            required command line arguments 'images' and 'out', where
            'images' is a path to a directory and 'out' is a path to a
            csv file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_images', type = str, required = True)
    parser.add_argument('-path_out_csv', type = str, required = True)
    parser.add_argument('-train_prop', type = float, default = 0.6)
    parser.add_argument('-val_prop', type = float, default = 0.2)
    parser.add_argument('-test_prop', type = float, default = 0.2)
    args = parser.parse_args()
    return args

def main():

    args = handle_args()

    images = load_images(args.path_images)
    patient_ids = np.unique([image.patient_id for image in images])
    split_ids = partition_ids(
        patient_ids, 
        args.train_prop, 
        args.val_prop, 
        args.test_prop
    )
    write_dict_to_csv(split_ids, args.path_out_csv)

if __name__ == '__main__':
    main()
