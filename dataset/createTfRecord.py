#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:55:58 2019

@author: siddharth
"""

import os



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
base_dataset_dir_aug_voc = '<pascal/augmented/VOC/images/path>/benchmark_RELEASE/dataset'
images_folder_name_aug_voc = 'img/'
annotations_folder_name_aug_voc = 'cls/'
images_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc,images_folder_name_aug_voc)
annotations_dir_aug_voc = os.path.join(base_dataset_dir_aug_voc,annotations_folder_name_aug_voc)


