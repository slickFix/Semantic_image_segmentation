#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:35:44 2019

@author: siddharth
"""

import os
import urllib
import tarfile

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