# Original Data Sets Used in this Project

All Stanford patient data is protected and cannot be made publicly available.

## prostate_convert_images

The original data set containing T2, DWI, and ADC images for patients with some having a prostate mask and all having lesion masks. Imon Banerjee uploaded ‘MLpatientNewTest.mat’ to box on 4/27/2018. Mia Kanzawa converted from MATLAB object to DICOM and uploaded to box on 5/14/2018.

## registered_images 

Image set with corresponding slice numbers registered using a shape-based registration algorithm. Imon Banerjee uploaded ‘Registered_data.mat’ to box on 7/27/2018. Mia Kanzawa converted from MATLAB object to DICOM and uploaded to box on 7/27/2018.

## prostate ROI

Supposed to be a set of images with patient IDs 202 to 268. I haven’t actually found any images in the folder. Imon Banerjee uploaded to box on 7/27/2018.

## New51016-prostate

Contains T2 prostate masks for patients with IDs 144 to 178. Includes prostate masks for some ADC and DWI slices. Does not include lesion masks. Imon Banerjee uploaded to box on 7/27/2018.

## t2_w_proi_flipped

Contains T2 prostate masks for images in registered data set that were missing them (patients with IDs 47 to 140). Created by Mia Kanzawa and uploaded to box on 9/11/2018.

## subset_needs_mask_dcm 

Contains prostate masks for the three patients whose largest T2 slice were missing masks. Created and uploaded to box by Mia Kanzawa on 11/16/2018.

## patient_labels.csv

Contains information about lesion class (benign vs. malignant) and gleason score (low vs. high).
