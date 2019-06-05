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

from metric import *

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

log_folder = './tensor_board_logs'

with open(log_folder+'/'+model_name+'/save/data.json','r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)    

# =============================================================================
# # list of classes
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
# 255=unknown

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

# =============================================================================
# # test data preprocessing
# =============================================================================

TEST_DATASET_DIR = './dataset/tfrecords'
TEST_FILE = 'test.tfrecords'

test_filenames = [os.path.join(TEST_DATASET_DIR,TEST_FILE)]
test_dataset = tf.data.TFRecordDataset(test_filenames)
test_dataset = test_dataset.map(tf_record_parser)  #parse the record into tensors
test_dataset = test_dataset.map(lambda image,annotation,image_shape:scale_image_with_crop_padding(image,annotation,image_shape,args.crop_size))
test_dataset = test_dataset.shuffle(buffer_size=500)
test_dataset = test_dataset.batch(args.batch_size)

# =============================================================================
# # creating dataset iterator 
# =============================================================================

iterator = test_dataset.make_one_shot_iterator()
batch_images_tf,batch_labels_tf,batch_shapes_tf = iterator.get_next()

# =============================================================================
# # forward propogation
# =============================================================================

logits_tf = neural_network.deeplab_v3(batch_images_tf,args,is_training=False,reuse=False)

valid_labels_batch_tf,valid_logits_batch_tf = training_util.get_valid_logits_and_labels(
        annotation_batch_tensor=batch_labels_tf,
        logits_batch_tensor=logits_tf,
        class_labels = class_labels)

# =============================================================================
# # loss definition
# =============================================================================

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=valid_labels_batch_tf,logits=valid_logits_batch_tf)
loss = tf.reduce_mean(cross_entropy)

tf.summary.scalar('test_loss',loss)

# =============================================================================
# # defining prediction and prediction probabilities
# =============================================================================

predictions_tf = tf.argmax(logits_tf,axis = 3)
probabilites_tf = tf.nn.softmax(logits_tf)


merged_summary_op = tf.summary.merge_all()
saver  = tf.train.Saver()

save_folder = os.path.join(log_folder,model_name,'save')

# =============================================================================
# # creating tf.Session()
# =============================================================================

with tf.Session() as sess:
    
    # initializing variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    # restore variables from disk
    saver.restore(sess,os.path.join(save_folder,'model.ckpt'))
    print("model ",model_name,' restored')
    
    mean_IoU = []
    mean_pixel_acc = []
    mean_freq_weighted_IU = [] 
    mean_acc = []
    
    while True:
        try:
            batch_images_np,batch_labels_np,batch_shapes_np,batch_predictions_np,tb_summary = \
                sess.run([batch_images_tf,batch_labels_tf,batch_shapes_tf,predictions_tf,merged_summary_op])
            
            heights,widths = batch_shapes_np
            
            
            for i in range(batch_predictions_np.shape[0]):            # looping through the images in the batch and extracting the valid areas from the tensors

                
                label_image = batch_labels_np[i]
                pred_image = batch_predictions_np[i]
                actual_image = batch_images_np[i]
                
                indices = np.where(label_image!=255)
                
                label_image = label_image[indices]
                pred_image = pred_image[indices]
                actual_image = actual_image[indices]
                
                if label_image.shape[0] == 513*513:
                    label_image = np.reshape(label_image,(513,513))
                    pred_image = np.reshape(pred_image,(513,513))
                    actual_image = np.reshape(actual_image,(513,513,3))
                    
                else:
                    label_image = np.reshape(label_image,(heights[i],widths[i]))
                    pred_image = np.reshape(pred_image,(heights[i],widths[i]))
                    actual_image = np.reshape(actual_image,(heights[i],widths[i],3))
                    
                pix_acc = pixel_accuracy(pred_image,label_image)
                m_acc = mean_accuracy(pred_image,label_image)
                IoU = mean_IU(pred_image,label_image)
                frequency_weighted_IU = frequency_weighted_IU(pred_image,label_image)
                
                
                mean_pixel_acc.append(pix_acc)
                mean_acc.append(m_acc)
                mean_IoU.append(IoU)
                mean_freq_weighted_IU.append(frequency_weighted_IU)
                
                
                f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(8,8))
                
                ax1.imshow(actual_image.astype(np.uint8))
                ax2.imshow(label_image)
                ax3.imshow(pred_image)
                plt.show()
                
        except tf.errors.OutOfRangeError:
            break
        
    print("Mean pixel accuracy:", np.mean(mean_pixel_acc))
    print("Mean accuraccy:", np.mean(mean_acc))
    print("Mean IoU:", np.mean(mean_IoU))
    print("Mean frequency weighted IU:", np.mean(mean_freq_weighted_IU))
                
            
            
    