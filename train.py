#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:01:51 2019

@author: siddharth
"""

import argparse
import tensorflow as tf
import numpy as np
import os
import json

# =============================================================================
# #list of classes for training
# =============================================================================

# 0=background
# 1=aeroplane
# 2=bicycle
# 3=bird
# 4=boat
# 5=bottle
# 6=bus
# 7=car
# 8=cat
# 9=chair
# 10=cow
# 11=diningtable
# 12=dog
# 13=horse
# 14=motorbike
# 15=person
# 16=potted plant
# 17=sheep
# 18=sofa
# 19=train
# 20=tv/monitor

# =============================================================================
# # defining argument parser
# =============================================================================

parser = argparse.ArgumentParser()

train_arg = parser.add_argument_group('Training params')

train_arg.add_argument('--batch_norm_epsilon',type = float,default = 1e-5,help = 'batch norm epsilon argument for batch normalisation')
train_arg.add_argument('--batch_norm_decay',type = float,default = 0.9997,help = 'batch norm decay argument for batch normalisation')
train_arg.add_argument('--number_of_classes',type = int,default = 21,help = 'number of classess to be predicted')
train_arg.add_argument('--l2_regularizer',type = float,default = 1e-4,help='l2 regularizer parameter')
train_arg.add_argument('--starting_learning_rate',type = float,default=1e-5,help='initial learning rate')
train_arg.add_argument('--multi_grid',type = list,default = [1,2,4],help = 'spatial pyramid pooling rates')
train_arg.add_argument('--output_stride',type=int,default=16,help='output stride of the network')
train_arg.add_argument('--gpu_id',type = int,default=0,help = 'id of the gpu to be used')
train_arg.add_argument('--crop_size',type = int,default=513,help='image cropsize')
train_arg.add_argument('--resnet_model',default = 'resnet_v2_50',choices = ['resnet_v2_50','resnet_v2_101','resnet_v2_152','resnet_v2_200'],help='resnet model to use as feature extractor.')
train_arg.add_argument('--batch_size',type = int,defautl=8,help = 'batch size for network to train')

environ_arg= parser.add_argument_group('Environment parameter')

environ_arg.add_argument('--current_best_val_loss',type=int,default=99999,help='best validation loss value')
environ_arg.add_argument('--accumulated_validation_miou',type=int,default=0,help = 'accumulated validation intersection over union')

args = parser.parse_arg()

