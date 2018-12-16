import argparse
from utils import *

def fuse_patient(images):
    t2_PROI = list_filter(lambda img: img.modality == 'T2' and img.typeof == 'PROI', images)[0]

    Slices = list_filter(lambda img: img.typeof == 'Slice', images)
    t2_Slice = list_filter(lambda img: img.modality == 'T2', Slices)[0]
    adc_Slice = list_filter(lambda img: img.modality == 'ADC', Slices)[0]
    dwi_Slice = list_filter(lambda img: img.modality == 'DWI', Slices)[0]
    
    ROIs = list_filter(lambda img: img.typeof == 'ROI', images)
    t2_ROI = list_filter(lambda img: img.modality == 'T2', ROIs)[0]
    adc_ROI = list_filter(lambda img: img.modality == 'ADC', ROIs)[0]
    dwi_ROI = list_filter(lambda img: img.modality == 'DWI', ROIs)[0]
    
    width, height = t2_Slice.pixel_array.shape
    
    Slices_fused_pixels = np.zeros((width, height, 3), dtype = 'uint8')
    Slices_fused_pixels[:,:,0] = t2_Slice.pixel_array
    Slices_fused_pixels[:,:,1] = adc_Slice.pixel_array
    Slices_fused_pixels[:,:,2] = dwi_Slice.pixel_array
    
    ROIs_fused_pixels = np.zeros((width, height, 3), dtype = 'uint8')
    ROIs_fused_pixels[:,:,0] = t2_ROI.pixel_array
    ROIs_fused_pixels[:,:,1] = adc_ROI.pixel_array
    ROIs_fused_pixels[:,:,2] = dwi_ROI.pixel_array
    
    PROI = Image(t2_PROI.name.replace('T2', 'FUSED'), t2_PROI.pixel_array)
    Slice_fused_image = Image(t2_Slice.name.replace('T2', 'FUSED'), Slices_fused_pixels)
    ROI_fused_image = Image(t2_ROI.name.replace('T2', 'FUSED'), ROIs_fused_pixels)
    
    return [PROI, Slice_fused_image, ROI_fused_image]

def fuse_images(images):
    patient_dict = split_by_patient(images)
    fused_patients = list_map(
        lambda patient_images: fuse_patient(patient_images),
        patient_dict.values()
    )
    return flatten(fused_patients)

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-images', type = str, required = True)
    parser.add_argument('-out', type = str, required = True)
    args = parser.parse_args()
    return args

def main():
    paths = handle_args()
    images = load_images(paths.images)
    images_fused = fuse_images(images)
    write_images(paths.out, images_fused)
    
if __name__ == '__main__':
    main()
