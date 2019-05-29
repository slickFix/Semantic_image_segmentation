#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:35:13 2019

@author: siddharth
"""

# importing libraries

import tensorflow as tf
print(" Tf version : ", tf.__version__)
import numpy as np
from matplotlib import pyplot as plt
import neural_network
slim = tf.contrib.slim
import os
import argparse
import json
from preprocessing.read_data import tf_record_parser,scale_image_with_crop_padding
from preprocessing import training_util

from metrics import *

plt.interactive(False)


# =============================================================================
# # defining argument parser
# =============================================================================
parser = argparse.ArgumentParser()

envrag = parser.add_argument_group("Eval params")
envrag.add_argument("--model_id",default=2019,type=int,help="Model id name  to be loaded ")
input_args = parser.parse_args()

model_name = str(input_args.model_id)


# =============================================================================
# # restoring train file arguments 
# =============================================================================

log_folder = './tboard_logs'

with open(log_folder+'/'+model_name+'/train/data.json','r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)    


