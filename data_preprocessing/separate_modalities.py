import argparse
from utils import *

def handle_args():
    """
    Args:
        None
    Returns:
        args (object): An object with attributes corresponding to the
            required command line arguments 'images', 't2', 'adc', and
            'dwi', which are all paths to directories
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-t2', type = str, required = True)
    parser.add_argument('-adc', type = str, required = True)
    parser.add_argument('-dwi', type = str, required = True)
    args = parser.parse_args()
    return args

def main():

    paths = handle_args()
    
    images = load_images(paths.images)
    images_dict = split_by_modality(images)
       
    write_images(paths.t2, images_dict['T2'])
    write_images(paths.adc, images_dict['ADC'])
    write_images(paths.dwi, images_dict['DWI'])

if __name__ == '__main__':
    main()
