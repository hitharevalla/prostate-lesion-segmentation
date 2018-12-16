# CNNs for Segmenting Prostate Lesions from Multiparametric MRIs

This project was completed for CS 230 Deep Learning at Stanford University. It aims to segment prostate lesions using multiparametric MRI exams in order to improve targeting for biopsy, which is the current diagnostic requirement. For patient privacy, none of the data or image outputs of the IPython notebooks are included in this repository.

## Prerequisites

Libraries needed to run this code from start to finish include: tensorflow, Keras, pandas, scikit-learn, scikit-image, cv2, pydicom, imageio, and matplotlib.

## Organization and Execution of Code

- **data_preprocessing**: Includes all of the code that was used for data pre-processing plus some IPython notebooks for image visualization. These files were implemented specifically for my data set, and the entire data pre-processing pipeline could be executed by running the shell script workflow.sh.
- **models**: Contains the code used to load data, specify model architectures, train the models, and generate predictions along with directories containing the IPython notebooks that show the training and testing results.
  - *model_definitions*: Includes the Keras models that were built for this project.
  - *vgg_segnet*, *vvg_bayesian_segnet*, and *vgg_unet_padded*: Contains results from training these models for 50 epochs on the various subsets of the data. Includes a subdirectory for each of the data subsets that the models were run on. Each subdirectory contains an IPython notebook that shows the training rand testing results. Notebooks could be run by just specifying the data locations and hyperparameters, then clicking restart and run all. All image output has been suppressed for patient privacy. Extra files generated during the training process (ex: stored weights, plots of loss and dice over training epochs, prediction images, and the model generated using plot_model were removed to save storage space on my laptop).
- **raw_data/stanford_data**: Data not included, but directory contains a readme describing all of the data that was shared with me.
- **data**: Directory not included, but would contain all of the output from running the shell script described in data_preprocessing.

## Acknowledgments

Divam Gupta's repository [image-segmentation-keras](https://github.com/divamgupta/image-segmentation-keras) was used as a starting point for the models developed in this project.
