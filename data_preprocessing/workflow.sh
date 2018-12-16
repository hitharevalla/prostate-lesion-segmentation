# Note: Starting conditions require that the paths to raw data are:
  # Original data: ../../raw_data/stanford_data/prostate_convert_images
  # Prostate masks radiologist: ../../raw_data/stanford_data/New51016-prostate
  # Prostate masks Mia first batch: ../../raw_data/stanford_data/t2_w_proi_flipped
  # Prostate masks Mia second batch: ../../raw_data/stanford_data/subset_needs_mask_dcm
  # Labels: ../../raw_data/stanford_data/patient_labels.csv

# Download the VGG16 pre-trained weights
wget 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
mv https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5 ../data/https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5

# Combines data from the above files into one folder of png images
python3 combine_stanford_data.py '../data/stanford_combined'

# Keeps the slice in each modality with the maximum lesion area for each patient
python3 subset_max_lesion_area.py \
  -images '../data/stanford_combined' \
  -out '../data/stanford_max_lesion_area'

# Splits the combined data by modality
python3 separate_modalities.py \
  -images '../data/stanford_combined' \
  -t2 '../data/stanford_t2' \
  -adc '../data/stanford_adc' \
  -dwi '../data/stanford_dwi'

# Register images based on lesion mask centroid and bounding box corners
# Allows scaling, rotation, and translation
python3 register_images.py \
  -images '../data/stanford_max_lesion_area' \
  -out '../data/stanford_max_lesion_area_registered'
  
# Creates fused versions of the max lesion area image sets
python3 fuse_images.py \
  -images '../data/stanford_max_lesion_area' \
  -out '../data/stanford_max_lesion_area_fused'
python3 fuse_images.py \
  -images '../data/stanford_max_lesion_area_registered' \
  -out '../data/stanford_max_lesion_area_registered_fused'

# Crop the images using PROIs
python3 crop_images.py \
  -images '../data/stanford_t2' \
  -out '../data/stanford_t2_cropped' \
  -padding 20
python3 crop_images.py \
  -images '../data/stanford_adc' \
  -out '../data/stanford_adc_cropped' \
  -padding 20
python3 crop_images.py \
  -images '../data/stanford_dwi' \
  -out '../data/stanford_dwi_cropped' \
  -padding 20
python3 crop_images.py \
  -images '../data/stanford_max_lesion_area_fused' \
  -out '../data/stanford_max_lesion_area_fused_cropped' \
  -padding 20
python3 crop_images.py \
  -images '../data/stanford_max_lesion_area_registered_fused' \
  -out '../data/stanford_max_lesion_area_registered_fused_cropped' \
  -padding 20 

# Splits the subject IDs into train, validation, and test sets
python3 split_ids.py \
  -path_images '../data/stanford_combined' \
  -path_out_csv '../data/stanford_train_val_test_split_ids.csv'

# Splits the data into train, validation, and test sets within images folder
# Each split has a Slice and ROI sub-folder
python3 split_images.py \
  -images '../data/stanford_t2_cropped' \
  -split_csv '../data/stanford_train_val_test_split_ids.csv'
python3 split_images.py \
  -images '../data/stanford_adc_cropped' \
  -split_csv '../data/stanford_train_val_test_split_ids.csv'
python3 split_images.py \
  -images '../data/stanford_dwi_cropped' \
  -split_csv '../data/stanford_train_val_test_split_ids.csv'
python3 split_images.py \
  -images '../data/stanford_max_lesion_area_fused_cropped' \
  -split_csv '../data/stanford_train_val_test_split_ids.csv'
python3 split_images.py \
  -images '../data/stanford_max_lesion_area_registered_fused_cropped' \
  -split_csv '../data/stanford_train_val_test_split_ids.csv'

# Augment the training images
python3 augment_images.py \
  -images '../data/stanford_t2_cropped/train/Slice' \
  -out '../data/stanford_t2_cropped/train/Slice' 
python3 augment_images.py \
  -images '../data/stanford_t2_cropped/train/ROI' \
  -out '../data/stanford_t2_cropped/train/ROI' 
  
python3 augment_images.py \
  -images '../data/stanford_adc_cropped/train/Slice' \
  -out '../data/stanford_adc_cropped/train/Slice' 
python3 augment_images.py \
  -images '../data/stanford_adc_cropped/train/ROI' \
  -out '../data/stanford_adc_cropped/train/ROI' 
  
python3 augment_images.py \
  -images '../data/stanford_dwi_cropped/train/Slice' \
  -out '../data/stanford_dwi_cropped/train/Slice' 
python3 augment_images.py \
  -images '../data/stanford_dwi_cropped/train/ROI' \
  -out '../data/stanford_dwi_cropped/train/ROI' 
  
python3 augment_images.py \
  -images '../data/stanford_max_lesion_area_fused_cropped/train/Slice' \
  -out '../data/stanford_max_lesion_area_fused_cropped/train/Slice' 
python3 augment_images.py \
  -images '../data/stanford_max_lesion_area_fused_cropped/train/ROI' \
  -out '../data/stanford_max_lesion_area_fused_cropped/train/ROI' 
  
python3 augment_images.py \
  -images '../data/stanford_max_lesion_area_registered_fused_cropped/train/Slice' \
  -out '../data/stanford_max_lesion_area_registered_fused_cropped/train/Slice' 
python3 augment_images.py \
  -images '../data/stanford_max_lesion_area_registered_fused_cropped/train/ROI' \
  -out '../data/stanford_max_lesion_area_registered_fused_cropped/train/ROI' 

