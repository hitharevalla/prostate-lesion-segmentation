from utils import *
import imageio
import numpy as np
import os
import pydicom
from skimage.transform import resize
import sys

def load_original_images(path_dir):
    """
    Args:
        path_dir (string): Path to directory containing the original data set
            of DICOM images with naming convention
            [patient ID]_[slice number]_[modality]_[Slice or ROI or PROI].dcm
    Returns:
        images_standardized (list): A list of Image objects from path_dir
    """
    file_names = list_filter(
        lambda file_name: file_name.endswith('.dcm'), 
        os.listdir(path_dir)
    )
    image_names = list_map(
        lambda file_name: file_name.strip('.dcm'),
        file_names
    )
    file_paths = list_map(
        lambda file_name: os.path.join(path_dir, file_name),
        file_names
    )
    pixel_arrays = list_map(
        lambda file_path: pydicom.read_file(file_path).pixel_array,
        file_paths
    )
    images = list_map(
        lambda name, pixels: Image(name, pixels),
        image_names, pixel_arrays
    )
    images_standardized = list_map(
        lambda image: standardize_image(image),
        images
    )

    return images_standardized

def load_patient(path_dir, modality, modality_dir, patient_id):
    """
    Args:
        path_dir (string): Path to folder containing T2, ADC, and DWI PROIs.
            See load_PROIs_radiologist for specifications
        modality (string): One of 'T2', 'ADC', or 'DWI'
        modality_dir (string): One of 'DISCOPEAK/DISCOPEAKROIimages',
            'ADC/ADCROIimages', or 'DWI/DWIROIimages'
        patient_id (string): The ID of the patient to load images for
    Returns:
        patient_modality_images_standardized (list): A list of Image objects 
            from the patient's modality directory
    """
    patient_path = os.path.join(path_dir, patient_id, modality_dir)
    file_names = list_filter(
        lambda file_name: file_name.endswith('.bmp'),
        sorted(os.listdir(patient_path))
    )
    image_names = list_map(
        lambda slice_num: '_'.join([patient_id, slice_num, modality, 'PROI']),
        list_map(str, range(1, len(file_names) + 1))
    )
    file_paths = list_map(
        lambda image_name: os.path.join(patient_path, image_name),
        file_names
    )
    pixel_arrays = list_map(imageio.imread, file_paths)
    patient_modality_images = list_map(
        lambda name, pixels: Image(name, pixels),
        image_names, pixel_arrays
    )
    patient_modality_images_standardized = list_map(
        lambda image: standardize_image(image),
        patient_modality_images
    )

    return patient_modality_images_standardized

def load_modality(path_dir, modality, modality_dir):
    """
    Args:
        path_dir (string): Path to folder containing T2, ADC, and DWI PROIs.
            See load_PROIs_radiologist for specifications
        modality (string): One of 'T2', 'ADC', or 'DWI'
        modality_dir (string): One of 'DISCOPEAK/DISCOPEAKROIimages',
            'ADC/ADCROIimages', or 'DWI/DWIROIimages'
    Returns:
        modality_images (list): A list of Image objects from the modality's
            directory
    """
    patient_ids = list_filter(
        lambda patient_id: os.path.isdir(os.path.join(path_dir, patient_id)),
        os.listdir(path_dir)
    )
    patient_images = list_map(
        lambda pt_id: load_patient(path_dir, modality, modality_dir, pt_id), 
        patient_ids
    )
    modality_images = [image for images in patient_images for image in images]

    return modality_images

def load_PROIs_radiologist(path_dir):
    """
    Args:
        path_dir (string): Path to folder containing T2, ADC, and DWI PROIs. 
            Should have folder containing T2 PROIs with structure
            folder_path/[patient ID]/DISCOPEAK/DISCOPEAKROIimages,
            folder containing ADC PROIs with structure
            folder_path/[patient ID]/ADC/ADCROIimages,
            and folder containing DWI PROIs with structure 
            folder_path/[patient ID]/DWI/DWIROIimages
            The number of PROIs in each folder should be same as number of 
            slices for the modality in the original data set. 
            File names should be of the format
            DISCOPEAKPeak_contrast_DISCOSlice[slice number].bmp, or 
            ADCApparent_Diffusion_Coefficient_(mm2s)Slice[slice number].bmp, 
            or DWIAx_DWI_prostateSlice[slice number].bmp where the slice
            numbers may not begin with 1, bute when sorted correspond to that
            ordering in the original data set
    Returns:
        images (list): A list of Image objects from path_dir
    """
    t2_images = load_modality(path_dir, 'T2', 'DISCOPEAK/DISCOPEAKROIimages')
    adc_images = load_modality(path_dir, 'ADC', 'ADC/ADCROIimages')
    dwi_images = load_modality(path_dir, 'DWI', 'DWI/DWIROIimages')

    images = t2_images + adc_images + dwi_images

    return images

def load_PROIs_mia(path_dir):
    """
    Args: 
        path_dir (string): Path to folder containing the T2 PROIs that were
            annotated by Mia. Folder contains DICOM images with naming format
            [patient ID]_[slice number]_[modality]_PROI_flipped.dcm
    Returns:
        images_standardized (list): A list of Image objects from path_dir
    """
    file_names = list_filter(
        lambda file_name: file_name.endswith('.dcm'),
        os.listdir(path_dir)
    )
    image_names = list_map(
        lambda file_name: file_name.strip('_flipped.dcm'),
        file_names
    )
    file_paths = list_map(
        lambda file_name: os.path.join(path_dir, file_name),
        file_names
    )
    pixel_arrays = list_map(
        lambda file_path: pydicom.read_file(file_path).pixel_array,
        file_paths
    )
    images = list_map(
        lambda name, pixels: Image(name, pixels),
        image_names, pixel_arrays
    )
    images_standardized = list_map(
        lambda image: standardize_image(image),
        images
    )

    return images_standardized

def load_raw_images():
    """
    Args:
        None
    Returns:
        images (list): A list of Image objects from all of the data
            directories that were loaded
    """
    path_original_data = '../raw_data/stanford_data/prostate_convert_images'
    path_PROIs_radiologist = '../raw_data/stanford_data/New51016-prostate'
    path_PROIs_mia_one = '../raw_data/stanford_data/t2_w_proi_flipped'
    path_PROIs_mia_two = '../raw_data/stanford_data/subset_needs_mask_dcm'

    original_images = load_original_images(path_original_data)
    radiologist_PROIs = load_PROIs_radiologist(path_PROIs_radiologist)
    mia_PROIs_one = load_PROIs_mia(path_PROIs_mia_one)
    mia_PROIs_two = load_PROIs_mia(path_PROIs_mia_two)

    images = original_images\
        + radiologist_PROIs\
        + mia_PROIs_one\
        + mia_PROIs_two\

    return images
 
def has_label(image):
    """
    Args:
        image (Image object): An image object that may or may not have a 
            diagnosis and gleason score
    Returns:
        (boolean): True if the patient has a diagnosis and gleason score, 
            otherwise False
    """
    return image.get_diagnosis() and image.get_gleason()
   
def filter_images(images):
    """
    Args:
        images (list): A list of Image objects
    Returns:
        images (list): A list of Image objects with bad images manually
            removed and images missing labels removed
    Note: The files named '49_2_ADC_ROI.dcm', '50_1_DWI_ROI.dcm', and 
        '50_2_DWI_ROI.dcm' are actually 'Slice' types, so they are removed
    """

    bad_images = ['50_1_DWI_ROI', '50_2_DWI_ROI', '49_2_ADC_ROI']

    images = list_filter(has_label, images)
    images = list_filter(
        lambda image: image.name not in bad_images,
        images
    )

    return images

def main():

    path_out_dir = sys.argv[1]

    images = load_raw_images()
    filtered_images = filter_images(images)
    
    write_images(path_out_dir, filtered_images)

if __name__ == '__main__':
    main()
