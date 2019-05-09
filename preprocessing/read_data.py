#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:35:44 2019

@author: siddharth
"""

import os
import urllib
import tarfile

import tensorflow as tf

def download_resnet_checkpoint_if_necessary(resnet_checkpoints_path,resnet_model_name):
    
    if not os.path.exists(resnet_checkpoints_path):
        
        # creating the path and downloading the checkpoints
        os.mkdir(resnet_checkpoints_path)
        
        filename = resnet_model_name+'_2017_04_14.tar.gz'
        
        url = "http://download.tensorflow.org/models/" + filename
        fullpath = os.path.join(resnet_checkpoints_path,filename)
        urllib.request.urlretrieve(url,fullpath)
        
        tarfile_ = tarfile.open(fullpath,"r:gz")
        tarfile_.extractall(path = resnet_checkpoints_path)
        tarfile_.close()
        
        print("Resnet checkpoints for model ",resnet_model_name," downloaded successfully.")
    else:
        print("Resnet checkpoints already exits")
        
        
def tf_record_parser(record):
    
    keys_to_features = {
            "image_raw":tf.FixedLenFeature((),tf.string,default_value=""),
            "annotation_raw":tf.FixedLenFeature([],tf.string),
            "height":tf.FixedLenFeature((),tf.int64),
            "width":tf.FixedLenFeature((),tf.int64)
            }
    
    features = tf.parse_single_example(record,keys_to_features)
    
    image  = tf.decode_raw(features['image_raw'],tf.uint8)
    annotation = tf.decode_raw(features['annotation_raw'],tf.uint8)
    
    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    
    # reshape input and annotation images
    
    image = tf.reshape(image,(height,width,3),name='image_rashape')
    annotation = tf.reshape(annotation,(height,width,1),name='annotation_reshape')
    annotation = tf.to_int32(annotation)
    
    return tf.to_float(image),annotation,(height,width)
    