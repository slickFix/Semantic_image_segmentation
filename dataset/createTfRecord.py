#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:55:58 2019

@author: siddharth
"""

import os
import requests
import tarfile
import numpy as np
import tensorflow as tf
import scipy.io as scio

from imageio import imread

# =============================================================================
# # Downloading dataset
# =============================================================================
dataset_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
download_filename = 'augmented_voc.tar.gz'

print("Downloading dataset to the current path")

r = requests.get(dataset_url,stream = True)

with open(download_filename,'wb') as f:    
    for chunk in r.iter_content(chunk_size=1024*1024):
        if chunk:            
            f.write(chunk)

# extracting dataset from tar file
donwload_file_dir = os.path.join('./',download_filename)        
tarfile_ = tarfile.open(donwload_file_dir)
tarfile_.extractall()
tarfile_.close()

# removing the downloaded file
os.remove(download_filename)

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

# =============================================================================
# # defining train and val variables
# =============================================================================

# shuffling array and separating 10% for validation
np.random.seed(2019)
np.random.shuffle(images_filename_list)
test_images_filename_list = images_filename_list[:int(0.05*len(images_filename_list))]
val_images_filename_list = images_filename_list[int(0.05*len(images_filename_list)):int(0.10*len(images_filename_list))]
train_images_filename_list = images_filename_list[int(0.10*len(images_filename_list)):]

TRAIN_DATASET_DIR = './tfrecords'
if not os.path.exists(TRAIN_DATASET_DIR):
    os.mkdir(TRAIN_DATASET_DIR)
    
TRAIN_FILE = 'train.tfrecords'
VAL_FILE = 'validation.tfrecords'
TEST_FILE = 'test.tfrecords'

train_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE))
val_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,VAL_FILE))
test_writer = tf.python_io.TFRecordWriter(os.path.join(TRAIN_DATASET_DIR,TEST_FILE))

# =============================================================================
# # defining utility functions
# =============================================================================
def _bytes_features(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list= tf.train.Int64List(value = [value]))

def read_annotation_from_mat_file(annotations_dir,images_name):
    annotaions_path = os.path.join(annotations_dir,(images_name.strip()+'.mat'))
    mat = scio.loadmat(annotaions_path)
    img = mat["GTcls"]['Segmentation'][0][0]
    return img

# =============================================================================
# # defining main function for dataset tfrecord format creation
# =============================================================================
def create_tfrecord_dataset(filename_list,writer):
    
    # create tfrecord
    read_imgs_counter = 0 
    for i,image_name in enumerate(filename_list):
        
        try:
            image_np = imread(os.path.join(images_dir_aug_voc,image_name.strip()+'.jpg'))
        
        except FileNotFoundError:
            try:
                # read from Pascal Voc path
                image_np = imread(os.path.join(images_dir_voc,image_name.strip()+'.jpg'))
            except FileNotFoundError:
                print("File image: ",image_name.strip()," not found!!")
                continue
        
        try:
            annotation_np = read_annotation_from_mat_file(annotations_dir_aug_voc,image_name)
        except FileNotFoundError:
            try:
                # read from Pascal VOC path
                annotation_np = imread(os.path.join(annotations_dir_voc,image_name.strip()+'.png'))
            
            except FileNotFoundError:
                print("File annotation: ",image_name.strip()," not found!!")
                continue
        
        read_imgs_counter+=1
        image_h = image_np.shape[0]
        image_w = image_np.shape[1]
        
        img_raw = image_np.tostring()
        annotation_raw = annotation_np.tostring()
        
        example = tf.train.Example(features=tf.train.Features(feature={
                'height':_int64_feature(image_h),
                'width':_int64_feature(image_w),
                'image_raw':_bytes_features(img_raw),
                'annotation_raw':_bytes_features(annotation_raw)}))
        
        writer.write(example.SerializeToString())
    
    print("End of TfRecord. Total images written: ",read_imgs_counter)
    writer.close()


# create training dataset
create_tfrecord_dataset(train_images_filename_list,train_writer)

# create validation dataset
create_tfrecord_dataset(val_images_filename_list,val_writer)

# create test dataset
create_tfrecord_dataset(test_images_filename_list,test_writer)
