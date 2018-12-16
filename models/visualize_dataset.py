import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_dataset(dir_images, dir_annotations, n_classes, dir_predictions = None, n_channel_annotations = 1):

    assert(dir_images[-1] == '/')
    assert(dir_annotations[-1] == '/')

    path_images = glob.glob(dir_images + '*.png')
    path_images.sort()
    
    path_annotations = glob.glob(dir_annotations + '*.png')
    path_annotations.sort()

    assert(len(path_images) == len(path_annotations))
    
    if n_classes == 2:
        colors = [(0,0,0), (255,255,255)]
    elif n_classes == 3:
        colors = [(0,0,0), (255,0,0), (0,255,0)]
    else:
        colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(n_classes)]
        
    assert(n_channel_annotations ==1 or n_channel_annotations == 3)
    if n_channel_annotations == 1:
        dim1, dim2, dim3 = [0] * 3
    if n_channel_annotations == 3:
        dim1, dim2, dim3 = range(3)
        
    if dir_predictions:
        
        assert(dir_predictions[-1] == '/')
        
        path_predictions = glob.glob(dir_predictions + '*.png')
        path_predictions.sort()
        
        assert(len(path_images) == len(path_predictions))
        
        for path_img, path_ann, path_pred in zip(path_images, path_annotations, path_predictions):
        
            assert(path_img.split('/')[-1] == path_ann.split('/')[-1] == path_pred.split('/')[-1])

            img = cv2.imread(path_img)
            ann = cv2.imread(path_ann)
            pred = cv2.imread(path_pred)

            ann_out = np.zeros_like(ann)
            pred_out = np.zeros_like(pred)

            for c in range(n_classes):
                
                ann_out[:,:,0] += ((ann[:,:,dim1] == c) * (colors[c][0])).astype('uint8')
                ann_out[:,:,1] += ((ann[:,:,dim2] == c) * (colors[c][1])).astype('uint8')
                ann_out[:,:,2] += ((ann[:,:,dim3] == c) * (colors[c][2])).astype('uint8')
                
                pred_out[:,:,0] += ((pred[:,:,dim1] == c) * (colors[c][0])).astype('uint8')
                pred_out[:,:,1] += ((pred[:,:,dim2] == c) * (colors[c][1])).astype('uint8')
                pred_out[:,:,2] += ((pred[:,:,dim3] == c) * (colors[c][2])).astype('uint8')

            f, axarr = plt.subplots(1,3)
            axarr[0].imshow(img)
            axarr[0].set_title('Slice')
            axarr[1].imshow(ann_out)
            axarr[1].set_title('ROI')
            axarr[2].imshow(pred_out)
            axarr[2].set_title('Prediction')
            f.suptitle(path_img.strip('.png').split('/')[-1])
            plt.show()

    else:
        
        for path_img, path_ann in zip(path_images, path_annotations):
        
            assert(path_img.split('/')[-1] == path_ann.split('/')[-1])

            img = cv2.imread(path_img)
            ann = cv2.imread(path_ann)
            
            ann_out = np.zeros_like(ann)

            for c in range(n_classes):
                ann_out[:,:,0] += ((ann[:,:,dim1] == c) * (colors[c][0])).astype('uint8')
                ann_out[:,:,1] += ((ann[:,:,dim2] == c) * (colors[c][1])).astype('uint8')
                ann_out[:,:,2] += ((ann[:,:,dim3] == c) * (colors[c][2])).astype('uint8')

            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(img)
            axarr[0].set_title('Slice')
            axarr[1].imshow(ann_out)
            axarr[1].set_title('ROI')
            f.suptitle(path_img.strip('.png').split('/')[-1])
            plt.show()