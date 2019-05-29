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
