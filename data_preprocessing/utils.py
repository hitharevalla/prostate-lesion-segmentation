import imageio
from itertools import groupby
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import SimpleITK as sitk
from skimage.measure import regionprops
from skimage.transform import resize

def flatten(list_of_lists):
    """
    Flattens a list of lists into a single list
    """
    return [item for list in list_of_lists for item in list]

def list_map(*args, **kwargs):
    """
    Like map but returns a list instead of a generator
    """
    return list(map(*args, **kwargs))

def list_filter(*args, **kwargs):
    """
    Like filter but returns a list instead of a generator
    """
    return list(filter(*args, **kwargs))

def initialize_dir(path_dir):
    """
    Creates the directory or if it already exists clears it
    Args:
        path_dir (string): The path to the directory to initialize
    Returns:
        None
    """
    if os.path.isdir(path_dir):
        shutil.rmtree(path_dir)
    os.makedirs(path_dir)

def load_labels():
    """
    Loads all recorded patient diagnoses and gleason scores
    Returns:
        labels (dataframe): A pandas dataframe indexed by patient ID that
            contains each patient's diagnosis and gleason score
    """
    path_labels = '../raw_data/stanford_data/patient_labels.csv'

    labels = pd.read_csv(
        path_labels,
        sep = ',',
        header = None,
        names = ['patient_id', 'diagnosis', 'gleason'],
        index_col = 'patient_id',
    )

    return labels

def write_images(path_out_dir, images,
        name_func = lambda image: image.name):
    """
    Args:
        path_out_dir (string): The path to write the images to as PNGs.
            If the directory is non-empty, all content will be deleted.
        images (list): A list of Image object
        name_func (function): A function that when given an image, returns 
            the appropriate name to use for writing the image as a string
    Returns:
        None
    """
    initialize_dir(path_out_dir)
    for image in images:
        imageio.imwrite(
            os.path.join(path_out_dir, name_func(image) + '.png'),
            image.pixel_array
        )

def load_images(path_dir):
    """
    Args:
        path_dir (string): The path from which images should be loaded.
            Images should be PNGs
    Returns:
        images (list): A list of Image objects from path_dir
    """
    file_names = list_filter(
        lambda file_name: file_name.endswith('.png'),
        os.listdir(path_dir)
    )
    image_names = list_map(
        lambda file_name: file_name.strip('.png'),
        file_names
    )
    file_paths = list_map(
        lambda file_name: os.path.join(path_dir, file_name),
        file_names
    )
    pixel_arrays = list_map(imageio.imread, file_paths)
    images = list_map(
        lambda name, pixels: Image(name, pixels),
        image_names, pixel_arrays
    )
    return images

class Image:
    """
    An object that holds an image and its retrievable data
    Args:
        image (numpy array): A numpy array that has been standardized
            to have size 256 x 256 and int8 pixel values
        image_name (string): The name of the image in the format
            [patient ID]_[slice number]_[modality]_[Slice or ROI or PROI]
    """
    def __init__(self, image_name, pixel_array):
        self.name = image_name
        self.pixel_array = pixel_array

    def __str__(self):
        return self.name

    @property
    def patient_id(self):
        return int(self.name.split('_')[0])

    @property
    def slice_num(self):
        return int(self.name.split('_')[1])

    @property
    def modality(self):
        return self.name.split('_')[2]

    @property
    def typeof(self):
        return self.name.split('_')[3]

    def get_diagnosis(self):
        labels = load_labels()
        try:
            patient_diagnosis = labels.loc[self.patient_id, 'diagnosis']
        except:
            patient_diagnosis = None
        return patient_diagnosis

    def get_gleason(self):
        labels = load_labels()
        try:
            patient_gleason = labels.loc[self.patient_id, 'gleason']
        except:
            patient_gleason = None
        return patient_gleason

    def get_lesion_area(self):
        if self.typeof != 'ROI':
            return None
        lesions = regionprops(self.pixel_array)
        lesion_area = np.sum([lesion.area for lesion in lesions])
        return lesion_area

    def get_partition(self, split_dict):
        return split_dict[str(self.patient_id)]

    def update_pixel_array(self, new_pixel_array):
        self.pixel_array = new_pixel_array
        return self
    
    def update_name(self, new_name):
        self.name = new_name
        return self

def split_by_modality(images):
    """
    Args:
        images (list): A list of Image objects
    Returns:
        image_dict (dict): A dictionary where the key is a modality and the 
        value is a list of images in each modality
    """
    images.sort(key = lambda image: image.modality)
    images_grouped = groupby(images, lambda image: image.modality)
    images_dict = dict((modal, list(imgs)) for modal, imgs in images_grouped)
    
    return images_dict

def split_by_patient(images):
    images.sort(key = lambda img: img.patient_id)
    images_grouped = groupby(
        images,
        lambda img: img.patient_id
    )
    images_grouped_dict = {group: list(images) for group, images in images_grouped}
    return images_grouped_dict

def get_associated_images(target, images):
    """
    Args:
        target (Image): An Image object for which we are searching for images
            with the same subject ID, slice number, and modality
        images (list): A list of Image objects to search in
    Returns:
        images (list): A list of Image objects with the same patient ID,
            slice number, and modality as the target image, excluding the
            target image
    """
    images = list_filter(
        lambda img: img.name.split('_')[0:3] == target.name.split('_')[0:3],
        images
    )
    images = list_filter(
        lambda img: img.typeof != target.typeof,
        images
    )
    return images

def standardize_image(image):
    """
    Args:
        image (Image): An image object
    Returns:
        image_standardized (Image): An Image with an updated pixelarray with 
            shape 224 by 224 and data type uint8. 'Slice' types have integer
            values ranging from 0 to 255, and 'ROI' and 'PROI' types have 
            values of either 0 or 1.
    """
    is_3d = len(image.pixel_array.shape) == 3
    
    if is_3d:
        new_size = (224, 224, 3)
    else:
        new_size = (224, 224)
    
    pixels = resize(
        image.pixel_array, 
        new_size,
        anti_aliasing = True,
        preserve_range = True
    )
    
    pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))

    if image.typeof in ['ROI', 'PROI']:
        pixels = pixels > 0.5
        pixels = pixels.astype('uint8')

    else:
        pixels = (pixels * 255).astype('uint8')

    image_standardized = image.update_pixel_array(pixels)

    return image_standardized

def display_patient_images(images):
    
    modalities = ['T2', 'ADC', 'DWI']
    typeofs = ['Slice', 'ROI', 'PROI']
    plots = list(product(modalities, typeofs))
    
    fig, axarr = plt.subplots(3, 3)
    for modality, typeof in plots:
        image = list_filter(
            lambda img: img.modality == modality and img.typeof == typeof,
            images
        )
        row = modalities.index(modality)
        col = typeofs.index(typeof)
        try:
            axarr[row, col].imshow(image[0].pixel_array, cmap='gray')
        except:
            dims = images[0].pixel_array.shape
            blank_image = np.ones(dims)
            axarr[row, col].imshow(blank_image, cmap='gray')
    
    x_labels = typeofs * 3
    y_labels = [modalities[0]] * 3 + [modalities[1]] * 3 + [modalities[2]] * 3
    for i, ax in enumerate(axarr.flat):
        ax.set(xlabel = x_labels[i], ylabel = y_labels[i])
        ax.label_outer()
    
    plt.show()
    
def display_images(images):
    patient_dict = split_by_patient(images)
    for patient_images in patient_dict.values():
        display_patient_images(patient_images)
