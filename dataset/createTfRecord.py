#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:55:58 2019

@author: siddharth
"""

import os
import numpy as np
import tensorflow as tf



# =============================================================================
# # defining base paths for pascal the original VOC dataset training images
# =============================================================================
base_dataset_dir_voc = '<path-to-voc-2012>/PascalVoc2012/train/VOC2012'
images_folder_name_voc = 'JPEGImages/'
annotations_folder_name_voc = 'SegmentationClass_1D/'
images_dir_voc = os.path.join(base_dataset_dir_voc,images_folder_name_voc)
annotations_dir_voc = os.path.join(base_dataset_dir_voc,annotations_folder_name_voc)


# =============================================================================
# # defining base paths for pascal augmented VOC images
# download: http://home.bharathh.info/pubs/codes/SBD/download.html
# =============================================================================
base_dataset_dir_aug_voc = './benchmark_RELEASE/dataset'
images_folder_name_aug_voc = 'img/'
annotations_folder_name_aug_voc = 'cls/'
images_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc,images_folder_name_aug_voc)
annotations_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc,annotations_folder_name_aug_voc)

# =============================================================================
# # defining function to get the list of all images saved in custom_train.txt
# =============================================================================
def get_files_list(filename):
    
    with open(filename,'r') as f:
        images_filename_list = [line.strip() for line in f.readlines()]
    return images_filename_list


images_filename_list = get_files_list('custom_train.txt')
print('Total number of training images',len(images_filename_list))
