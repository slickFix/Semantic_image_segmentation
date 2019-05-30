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
import neural_network

from preprocessing.read_data import download_resnet_checkpoint_if_necessary,\
                                    tf_record_parser,rescale_image_and_annotation_by_factor,\
                                    distort_randomly_image_color,random_flip_image_and_annotation,\
                                    scale_image_with_crop_padding
                                    
from preprocessing import training_util


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
# 255 = unknown

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

# =============================================================================
# # defining global varialbes
# =============================================================================

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
crop_size= args.crop_size

LOG_FOLDER = './tensor_board_logs'
TRAIN_DATASET_DIR = './dataset/tfrecords'
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'

resnet_checkpoints_path = './resnet/checkpoints/'
download_resnet_checkpoint_if_necessary(resnet_checkpoints_path, args.resnet_model)



# =============================================================================
# # defining training and validation dataset
# =============================================================================

training_filenames = [os.path.join(TRAIN_DATASET_DIR,TRAIN_FILE)]
training_dataset = tf.data.TFRecordDataset(training_filenames)
training_dataset = training_dataset.map(tf_record_parser) # Parse the record in tensors
training_dataset = training_dataset.map(rescale_image_and_annotation_by_factor)
training_dataset = training_dataset.map(distort_randomly_image_color)
training_dataset = training_dataset.map(lambda image,annotation,image_shape:scale_image_with_crop_padding(image,annotation,image_shape,crop_size))
training_dataset = training_dataset.map(random_flip_image_and_annotation) 
training_dataset = training_dataset.repeat()  # no of epochs (no values means inf time repeat)
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(args.batch_size)

validation_filenames = [os.path.join(TRAIN_DATASET_DIR,VALIDATION_FILE)]
validation_dataset = tf.data.TFRecordDataset(validation_filenames)
validation_dataset = validation_dataset.map(tf_record_parser) # Parse the record in tensors
validation_dataset = validation_dataset.map(lambda image,annotation,image_shape:scale_image_with_crop_padding(image,annotation,image_shape,crop_size))
validation_dataset = validation_dataset.shuffle(buffer_size = 100)
validation_dataset = validation_dataset.batch(args.batch_size)

class_labels = [ v for v in range(args.number_of_classes+1)]
class_labels[-1] = 255


# =============================================================================
# # defining dataset Iterators
# =============================================================================

handle = tf.placeholder(tf.string,shape=[])

iterator = tf.data.Iterator.from_string_handle(handle,training_dataset.output_types,training_dataset.output_shapes)

batch_images_tf,batch_labels_tf,_ = iterator.get_next()

training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# =============================================================================
# # =============================================================================
# # # =============================================================================
# # # # Forward Propogation
# # # =============================================================================
# # =============================================================================
# =============================================================================

is_training_tf = tf.placeholder(tf.bool,shape=[])

# logits dimension -> (batch_size, width, height, num_classes)
logits_tf = tf.cond(is_training_tf,true_fn=lambda:neural_network.deeplab_v3(batch_images_tf,args,is_training=True,reuse = False),
                    false_fn=lambda:neural_network.deeplab_v3(batch_images_tf,args,is_training=False,reuse=True))


# get valid logits and labels (factor the 255 padded mask for cross enetropy)
valid_labels_batch_tf,valid_logits_batch_tf = training_util.get_valid_logits_and_labels(
        annotation_batch_tensor= batch_labels_tf,
        logits_batch_tensor = logits_tf,
        class_labels=class_labels)


# =============================================================================
# # loss
# =============================================================================
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=valid_logits_batch_tf,labels=valid_labels_batch_tf)
loss = tf.reduce_mean(cross_entropy)

tf.summary.scalar('loss',loss)

# =============================================================================
# # prediction
# =============================================================================

prediction_tf = tf.argmax(logits_tf,axis=3)

# =============================================================================
# # optimizer
# =============================================================================

with tf.variable_scope('optimizer_vars'):
    global_step = tf.Variable(0,trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    train_step = tf.contrib.slim.learning.create_train_op(loss,optimizer,global_step=global_step)
    
    
# =============================================================================
# # Accuracy calculation    
# =============================================================================

miou,update_op = tf.contrib.metrics.streaming_mean_iou(tf.arg_max(valid_logits_batch_tf,axis=1),
                                                       tf.arg_max(valid_labels_batch_tf,axis=1),
                                                       num_classes=args.number_of_classes)

tf.summary.scalar('miou',miou)

# =============================================================================
# # Tensorboard log dir, saver and resnet model restore
# =============================================================================

# merging tensorboard summaries
merged_summary_op = tf.summary.merge_all()

process_str_pid = str(os.getpgid())
LOG_FOLDER = os.path.join(LOG_FOLDER,process_str_pid)

# creating tensorboard log folder if it doesn't exist
if not os.path.exists(LOG_FOLDER):
    print('Creating tensorboard log folder : ',LOG_FOLDER)
    os.mkdirs(LOG_FOLDER)
    
# creating restorer of (simple) resnet model and saver of current model
variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=[args.resnet_model + "/logits", "optimizer_vars",
                                                              "DeepLab_v3/ASPP_layer", "DeepLab_v3/logits"])

restorer = tf.train.Saver(variables_to_restore)    
saver = tf.train.Saver()

current_best_val_loss = np.inf


# =============================================================================
# # creating tf session
# =============================================================================

with tf.Session() as sess:
    
    # creating tb summary writer
    train_writer = tf.summary.FileWriter(LOG_FOLDER+'/train',sess.graph)
    val_writer = tf.summary.FileWriter(LOG_FOLDER+'/val')
    
    # initializing variables
    sess.run(tf.global_variables_initializer)
    sess.run(tf.local_variables_initializer)
    
    # loading resnet checkpoints
    try:
        restorer.restore(sess,'./resnet/checkpoints/'+args.resnet_model+'.ckpt')
        print("Model checkpoints for "+args.resnet_model+" restored !!")
        
    except FileNotFoundError:
        print("Resnet checkpoints not found. Please download ")
        
    
    # getting the string handles for different datasets 
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    
    
    #initialising training_iterator
    sess.run(training_iterator.initializer)
    
    # variables declaration
    validation_running_loss = []
    
    train_steps_before_eval =100
    validation_steps = 20
    
    while True:
        
        training_average_loss = 0
        
        # training loop
        for i in range(train_steps_before_eval): # no of batches run before evaluation
            _,global_step_val,train_loss,tb_summary = sess.run([train_step,global_step,loss,merged_summary_op],\
                                                               feed_dict={is_training_tf:True,\
                                                                          handle:training_handle})
            training_average_loss+=train_loss
            
            if i%10==0:
                train_writer.add_summary(tb_summary,global_step=global_step_val)
                
        training_average_loss /= train_steps_before_eval
        
        
        # at the end of each train interval, running validation
        sess.run(validation_iterator.initializer)
        
        validation_average_loss = 0
        validation_average_miou = 0
        
        # validation loop
        for i in range(validation_steps):
            
            val_loss,tb_summary, _ = sess.run([loss,merged_summary_op,update_op],
                                              feed_dict={handle:validation_handle,
                                                         is_training_tf:False})
            
            validation_average_loss+=val_loss
            validation_average_miou+=sess.run(miou)
            
        validation_average_loss/=validation_steps
        validation_average_miou/=validation_steps
        
        # Finding the global validation average loss
        validation_running_loss.append(validation_average_loss)
        validation_global_loss = np.mean(validation_running_loss)        
        
        val_writer.add_summary(tb_summary,global_step=global_step_val)
        
        
        # saving the model parameters if validation_global_loss < current_best_val_loss
        if validation_global_loss<current_best_val_loss:
            
            # saving the variables to the disk
            save_path = saver.save(sess,LOG_FOLDER+'/save'+'/model.ckpt')
            
            print("Model checkpoint written! Best average loss: ",validation_global_loss)
            
            # updating the metadata and saving it
            current_best_val_loss = validation_global_loss

            args.current_best_val_loss = str(current_best_val_loss)
        
            with open(LOG_FOLDER+'/save/'+'data.json','w') as fp:
                json.dump(args.__dict__,fp,sort_keys=True,indent=4)
                
        
        print('Global step : ',global_step_val,'\tAverage train loss : ',training_average_loss,'\tGlobal Validation avg loss : ',validation_global_loss,'\tMIoU : ',validation_average_miou)
        
    train_writer.close()
            
    
      
    
    
            
    