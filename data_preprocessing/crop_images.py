import argparse
import math
from skimage.measure import regionprops
from utils import *

def pad_to_square(image):
    """
    Args:
        image (Image): An image object
    Returns:
        image (Image): An square image padded with zeros on its shorter side
    """   
    is_3d = len(image.pixel_array.shape) == 3
    
    if is_3d:
        height, width, n_channels = image.pixel_array.shape
    else:
        height, width = image.pixel_array.shape

    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0

    if height > width:
        pad_left = math.ceil((height - width) / 2.)
        pad_right = math.floor((height - width) / 2.)

    if height < width:
        pad_top = math.ceil((width - height) / 2.)
        pad_bottom = math.floor((width - height) / 2.)
        
    if is_3d:
        pad_tuple = ((pad_top, pad_bottom), (pad_left, pad_right), (0,0))
    else:
        pad_tuple = ((pad_top, pad_bottom), (pad_left, pad_right))
    
    square_pixels = np.pad(
        image.pixel_array, 
        pad_tuple, 
        mode='constant'
    )

    square_image = image.update_pixel_array(square_pixels)

    return square_image

def crop_bbox(image, min_row, min_col, max_row, max_col, padding = 0):
    """
    Args:
        image (Image): An Image object
        min_row (int): Row (inclusive) defining upper bound of cropped image
        min_col (int): Col (inclusive) defining left bound of cropped image
        max_row (int): Row (exclusive) defining lower bound of cropped image
        max_col (int): Col (exclusive) defining right bound of cropped image
        padding (int): The number of pixels to pad the image by on any side,
            this must not force min_row, min_col, max_row, or max_col to 
            extend outside the bounds of the image
    Returns
        image (Image): A cropped image object with optional padding
    """
    min_row = min_row - padding
    min_col = min_col - padding
    max_row = max_row + padding
    max_col = max_col + padding
    
    cropped_pixels = image.pixel_array[min_row:max_row, min_col:max_col]
    cropped_image = image.update_pixel_array(cropped_pixels)
    return cropped_image

def crop_image_set_PROI(image_set, pad):
    """
    Args:
        image_set (list): A list of three image objects of types 'Slice', 
            'ROI', and 'PROI' with the same patient_id, slice_num, and modality
        pad (int): The number of pixels to pad the prostate region by
    Returns:
        image_set_cropped (list): The list of image objects that have been
            cropped based on the bounding box of the PROI and optional padding
    """
    Slice = list_filter(lambda i: i.typeof == 'Slice', image_set)[0]
    ROI = list_filter(lambda i: i.typeof == 'ROI', image_set)[0]
    PROI = list_filter(lambda i: i.typeof == 'PROI', image_set)[0]

    min_row, min_col, max_row, max_col = regionprops(PROI.pixel_array)[0].bbox

    Slice_cropped = crop_bbox(Slice, min_row, min_col, max_row, max_col, pad)
    ROI_cropped = crop_bbox(ROI, min_row, min_col, max_row, max_col, pad)
    PROI_cropped = crop_bbox(PROI, min_row, min_col, max_row, max_col, pad)

    image_set_cropped = [Slice_cropped, ROI_cropped, PROI_cropped]
    return image_set_cropped
    
def crop_all_images(images, pad):
    """
    Args:
        images (list): A list of image objects
        pad (int): The number of pixels to pad the prostate region by
    Returns
        images_cropped (list): A list of image objects where only the complete
            sets (having a Slice, ROI, and PROI with the same patient_id,
            slice_num, and modality) are all cropped around the bounding box
            of their associated PROI
    """
    images_cropped = []
    modality_dict = split_by_modality(images)
    for image_modal in modality_dict.values():
        images_Slice = list_filter(
            lambda img: img.typeof == 'Slice',
            image_modal
        )
        images_assoc = list_map(
            lambda img: get_associated_images(img, image_modal),
            images_Slice
        )
        images_sets = list_map(
            lambda Slice, imgs_assoc: imgs_assoc + [Slice],
            images_Slice, images_assoc
        )
        images_sets_complete = list_filter(
            lambda img_set: len(img_set) == 3,
            images_sets
        )
        images_modal_cropped = list_map(
            lambda img_set: crop_image_set_PROI(img_set, pad),
            images_sets_complete,
        )
        images_cropped.append(flatten(images_modal_cropped))

    return flatten(images_cropped)

def handle_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-out', type = str, required = True)
    parser.add_argument('-padding', type = int, default = 10)
    args = parser.parse_args()
    return args

def main():
    
    args = handle_args()

    images = load_images(args.images)
    cropped_images = crop_all_images(images, args.padding)
    cropped_images = list_map(
        lambda image: pad_to_square(image),
        cropped_images
    )
    cropped_images = list_map(
        lambda image: standardize_image(image),
        cropped_images
    )
    write_images(args.out, cropped_images)

if __name__ == '__main__':
    main()
