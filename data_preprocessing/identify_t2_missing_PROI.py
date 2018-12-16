import argparse
from utils import *

def identify_missing_PROIs(images, images_needing_PROI):
    """
    Args:
        images (list): A list of all Image objects to search for PROIs in
        images_needing_PROI (list): A list of Image objects that should
            have an associated PROI
    Returns:
        images_missing_PROI (list): The subset of images_needing_PROI that
            do not have an associated PROI in images
    """
    associated_types = list_map(
        lambda image: [i.typeof for i in get_associated_images(image, images)],
        images_needing_PROI
    )
    associated_types = dict(zip(images_needing_PROI, associated_types))
    images_missing_PROI = list_filter(
        lambda image: 'PROI' not in associated_types[image],
        images_needing_PROI
    )
    return images_missing_PROI

def handle_args():
    """
    Args:
        None
    Returns: 
        args (object): An object with attributes corresponding to the
            required command line arguments 'images' and 'out', which are
            both paths to directories
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-out', type = str, required = True)
    args = parser.parse_args()
    return args 

def main():
    
    paths = handle_args()

    images = load_images(paths.images)
    images_t2 = [img for img in images if img.modality == 'T2']
    images_t2_Slice = [img for img in images_t2 if img.typeof == 'Slice']
    images_t2_missing_PROI = identify_missing_PROIs(images, images_t2_Slice)

    write_images(paths.out, images_t2_missing_PROI)

if __name__ == '__main__':
    main()
