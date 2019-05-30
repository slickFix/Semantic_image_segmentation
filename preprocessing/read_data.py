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

from preprocessing.inception_preprocessing import apply_with_random_selector,distort_color


def random_flip_image_and_annotation(image_tensor,annotation_tensor,image_shape):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
    The function performs random flip of image and annotation tensors with probability of 1/2
    The flip is performed or not performed for image and annotation consistently, so that
    annotation matches the image"""
    
    original_shape = tf.shape(annotation_tensor)
    # ensure the annotation tensor has shape (width,height,1)
    annotation_tensor = tf.cond(tf.rank(annotation_tensor)<3,
                                lambda:tf.expand_dims(annotation_tensor,axis=2),
                                lambda:annotation_tensor)
    
    random_var = tf.random_uniform(maxval=2,dtype=tf.int32,shape=[])
    
    randomly_flipped_img = tf.cond(pred=tf.equal(random_var,0),
                                   true_fn=lambda:tf.image.flip_left_right(image_tensor),
                                   false_fn=lambda:image_tensor)
    
    randomly_flipped_ann = tf.cond(pred=tf.equal(random_var,0),
                                   true_fn=lambda:tf.image.flip_left_right(annotation_tensor),
                                   false_fn=lambda:annotation_tensor)
    
    return randomly_flipped_img,tf.reshape(randomly_flipped_ann, original_shape, name="reshape_random_flip_image_and_annotation"),image_shape

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

def rescale_image_and_annotation_by_factor(image,annotation,image_shape,min_scale = 0.5,max_scale=2):
    
    
    # data augmentation is done by randomly scaling the input images (from 0.5 to 2) during training
    
    input_shape = tf.shape(image)[0:2]
    input_shape_float = tf.to_float(input_shape)
    
    
    scale = tf.random_uniform(shape = [1],minval=min_scale,maxval=max_scale)
    
    scaled_input_shape = tf.to_int32(tf.round(input_shape_float*scale))
    
    image =tf.image.resize_images(image,scaled_input_shape,method=tf.image.ResizeMethod.BILINEAR)
    
    # use nearest neighbour for annotations resizeing in order to keep proper values
    
    annotation = tf.image.resize_images(annotation,scaled_input_shape,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image,annotation,image_shape
    
def distort_randomly_image_color(image_tensor,annotation_tensor,image_shape):
    
    # Accepts image tensor(w,h,3) and returns color distorted image.
    # Performs random brightness, saturation, hue, contrast change as it is performed for inception model training in TF-slim.
    # All the parameters of random variables are originally preserved.
    # Works in slow and fast mode. Slow mode performs only saturation and brightness random change.
    
    # returns img_float_distorted_original_range: Tensor of size(width,height,3) of type tf.float
    #         Image tensor with distorted color in [0,255] intensity range
    
    
    fast_mode = False
    
    # Normalizing the image 
    img_float_zero_one_range = tf.to_float(image_tensor)/255
    
    # Randomly distort the color of image. There are 4 ways to do it.
    # Credit: TF-Slim
    # https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py#L224
    # Most probably the inception models were trainined using this color augmentation:
    # https://github.com/tensorflow/models/tree/master/slim#pre-trained-models
    
    
    distorted_image = apply_with_random_selector(img_float_zero_one_range,
                                                 lambda x,ordering: distort_color(x,ordering,fast_mode=fast_mode),num_cases=4)
    
    img_float_distorted_original_range = distorted_image * 255
    
    return img_float_distorted_original_range,annotation_tensor,image_shape
    

def scale_image_with_crop_padding(image,annotation,image_shape,crop_size):
    
    image_croped = tf.image.resize_image_with_crop_or_pad(image,crop_size,crop_size)
    
    
    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    
    
    annotation_shifted_classes = annotation + 1
    
    cropped_padded_annotation = tf.image.resize_image_with_crop_or_pad(annotation_shifted_classes,crop_size,crop_size)
    
    mask_out_number = 255
    
    annotation_additional_mask_out = tf.to_int32(tf.equal(cropped_padded_annotation,0))*(mask_out_number+1)
    
    cropped_padded_annotation = cropped_padded_annotation+annotation_additional_mask_out-1
    
    return image_croped,tf.squeeze(cropped_padded_annotation),image_shape
