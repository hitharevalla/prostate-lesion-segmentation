import argparse
from utils import *
import numpy as np

def modality_subset_max_lesion_area(modality, patient_images):
    """
    Args:
        modality (string): The modality to find the largest lesion slice for
        patient_images (list): A list of Image objects for a single patient
    Returns:
        modality_images_subset (list): A list of images for a particular
            patient and modality where the images are for the slice with
            the largest lesion area
    Note: If no slices have a 'Slice' and 'ROI' image this returns None
    """
    modality_images = list_filter(
        lambda image: image.modality == modality,
        patient_images
    )
    slice_nums = np.unique([image.slice_num for image in modality_images])

    max_lesion_area = 0
    max_lesion_area_slice_num = None

    for slice_num in slice_nums:

        slice_Slice_image = list_filter(
            lambda im: im.slice_num == slice_num and im.typeof == 'Slice',
            modality_images
        )
        slice_ROI_image = list_filter(
            lambda im: im.slice_num == slice_num and im.typeof == 'ROI',
            modality_images
        )

        if not slice_Slice_image or not slice_ROI_image:
            continue

        lesion_area = slice_ROI_image[0].get_lesion_area()
        if lesion_area > max_lesion_area:
            max_lesion_area = lesion_area
            max_lesion_area_slice_num = slice_num

    if max_lesion_area_slice_num == None:
        return None

    modality_images_subset = list_filter(
        lambda image: image.slice_num == max_lesion_area_slice_num,
        modality_images
    )
    return modality_images_subset

def patient_subset_max_lesion_area(patient_id, images):
    """
    Args:
        patient_id (int): The ID of the patient whose images to subset
        images (list): A list of all Images objects in the data set
    Returns:
        patient_subset (list): A list of Image objects for a single 
            patient that only includes the slice with the largest lesion area 
            for each modality
    Note: If any modality is lacking a valid slice this returns None
    """
    modalities = np.unique([image.modality for image in images])
    patient_images = list_filter(
        lambda image: image.patient_id == patient_id,
        images
    )
    patient_images_subset = list_map(
        lambda modal: modality_subset_max_lesion_area(modal, patient_images),
        modalities
    )
    if None in patient_images_subset:
        return None
    return flatten(patient_images_subset)

def subset_max_lesion_area(images):
    """
    Args:
        images (list): A list of Image objects
    Returns:
        images_subset (list): A list of Image objects where each patient has 
            only one slice per modality, where the slice chosen is the one 
            with the largest lesion area
    """
    patient_ids = np.unique([image.patient_id for image in images])
    patient_subsets = list_map(
        lambda patient_id: patient_subset_max_lesion_area(patient_id, images),
        patient_ids
    )
    patient_subsets = list_filter(
        lambda patient_subset: patient_subset != None,
        patient_subsets
    )
    images_subset = flatten(patient_subsets)
    return images_subset

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
    subset_images = subset_max_lesion_area(images)
    write_images(paths.out, subset_images)

if __name__ == '__main__':
    main()

