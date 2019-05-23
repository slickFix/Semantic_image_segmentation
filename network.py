#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:20:50 2019

@author: siddharth
"""

import tensorflow as tf
slim = tf.contrib.slim

from resnet import resnet_v2,resnet_utils

# ImageNet mean statistics
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def deeplab_v3(inputs,args,is_training,reuse):
    
    
    # mean subtraction normalisation
    inputs = inputs-[_R_MEAN,_G_MEAN,_B_MEAN]
    
    # inputs shape = original shape [batch,513,513,3]
    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regualrizer,is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        resnet = getattr(resnet_v2,args.resnet_model)
        _,end_points = resnet(inputs,args.number_of_classes,
                              is_training=is_training,
                              global_pool = False,
                              output_stride = args.output_stride,
                              reuse = reuse)
        
        with tf.variable_scope("DeepLab_v3",reuse = reuse):
            
            # get block 4 feature outputs 
            net  = end_points[args.resnet_model+'/block4']
            
            net = atrous_spatial_pyramid_pooling(net,'ASPP_layer',depth=256,reuse=reuse)
            
            net = slim.conv2d(net,args.number_of_classes,[1,1],activation_fn=None,
                              normalizer_fn=None,scope = 'logits')
            
            size = tf.shape(inputs)[1:3]
            
            #resizing the output logits to match the labels dimensions
            net = tf.image.resize_bilinear(net,size)
            
            return net