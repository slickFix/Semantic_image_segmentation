#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:18:19 2019

@author: siddharth
"""

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""




import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

def distort_color(image,color_ordering=0,fast_mode = True,scope = None):
    
    """ Distort the color of a Tensor Image.
    
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather than adding that level of complication, we select a distinct ordering of color ops
    for each preprocessing thread.
    
    Args:
        image: 3-D Tensor containing a single image [0,1]
        color_ordering: Python int, a type of distortion(valid valus=0-3)
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        score: Optional scope for name_scope
    
    Returns:
        3-D tensor color distorted image on range[0,1]
        
    Raises:
        ValueError: if color_ordering not in [0-3]
    """
    
    
    with tf.name_scope(scope,'distort_color',[image]):
        if fast_mode:
            if color_ordering==0:
                image = tf.image.random_brightness(image,max_delta=32./255.)
                image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
          if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
          elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
          elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
          elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
          else:
            raise ValueError('color_ordering must be in [0, 3]')
            
     # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def apply_with_random_selector(x,func,num_cases):
    
    """ Computes func(x,sel) with sel sampled from [0...num_cases-1]
    
    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.
    
    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    
    sel = tf.random_uniform([],maxval=num_cases,dtype=tf.int32)
    
    # Passing real x to only one of the 'func' calls 
    
    return control_flow_ops.merge([func(control_flow_ops.switch(x,tf.equal(sel,case))[1],case) for case in range(num_cases)])[0]