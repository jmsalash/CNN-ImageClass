#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:34:54 2017

@author: JoseMa
"""

import tensorflow as tf
import numpy as np
import os
#from tensorflow.examples.tutorials.mnist import input_data
from skimage import io
import dataset
import sys
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import filters

from sklearn import preprocessing, model_selection
from sklearn.metrics import confusion_matrix
from datetime import datetime

import time



print ('Start time: ' + str(datetime.now().time()))

###############################
# PARAMETERS
###############################

# Image size
org_width = 800
org_height = 240
crop_w = 0.08
crop_h = 0.01   
resize_img_height = 112#120
ratio = int(org_width/org_height)
resize_img_width = resize_img_height*ratio#400

# Dataset
classes = ['Lanciafino10mBianca', 'Mototopo','VaporettoACTV']
n_classes = len(classes)
train_path="/Users/JoseMa/Documents/Uni/MARR/ML/2class"
validation_size = 0.05
# Number of wrong images to show after validation
val_samples = 15

# CNN Configuration
#iterations = 1000
epochs = 3
batch_size = 8
keep_probab = 0.8
features = 32
final_layer = 1024
# This can only be changed if code is adjusted!!!
num_channels = 1
num_pools = 2


# Create dataset. 
X, y = dataset6.prepare_dataset(train_path, classes)

# Turn classes into one hot vectors
Y_onehot = preprocessing.OneHotEncoder(sparse=False).fit_transform(y.reshape(-1,1)) 

# Split the dataset into train/test using sklearn, with the same proportion of samples per class
# Shuffle the training dataset before learning it.
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y_onehot, test_size=validation_size, stratify=y, shuffle=True, random_state=5)

#calculate number of iterations to complete the desired epochs
iterations = int((np.size(X_train,0)/batch_size)*epochs)
print('Number of iterations: ' + str(iterations))
print('Batch size: ' + str(batch_size))

#Show an example of how the image looks
imageTemp, labelTemp = dataset6.load_images(X_train[0:1], Y_train[0:1], resize_img_height, resize_img_width, crop_h, crop_w)
image = imageTemp[0]
img_width = image.shape[1]
img_height = image.shape[0]
out_height = img_height/pow(2,num_pools)
out_width = img_width/pow(2,num_pools)
train_size = len(X_train)


print(image.shape)

#show image
io.imshow(image)
io.show()



# given the size of the kernel, we initialise the variable(s)
# given the shape of the filter, initialise with randomly distributed values with a stddev of 0.1
def weight_variable(myName, shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial, name=myName)

# for bias, initial values will be constant
def bias_variable(myName, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=myName)

# input, kernel and shape of the kernel, strides (batch size, height, width, channels). Output same size of the input.
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# kernel of 2, horizontal and vertical. stride of 2. Never takes the same pixel more than once
# stride: we take all batches and all depths, but we skip one every 2 pixels [batch,pixel, pixel, layer]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 28x28 = 784
# 'None' means that it will change
x = tf.placeholder(tf.float32, shape=[None, img_height, img_width])
t = tf.placeholder(tf.float32, shape=[None, n_classes])

with tf.name_scope('convolutional_1'):
    W_conv1 = weight_variable('W_conv1',[5, 5, 1, features]) # 
    b_conv1 = bias_variable('b_conv1',[features])
    print('Convolutional Layer 1 - Shape: W1:{}, b1:{}'.format(W_conv1.get_shape(), b_conv1.get_shape()))

    
    # transform the image to correct format
    x_image = tf.reshape(x, [-1, img_height, img_width, 1])
    
    # 2d convolution of the image, using relu
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    # after that apply max pool to the result
    h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope('convolutional_2'):
    W_conv2 = weight_variable('W_conv2',[5, 5, features, features*2])
    b_conv2 = bias_variable('b_conv2',[features*2])
    print('Convolutional Layer 2 - Shape: W2:{}, b2:{}'.format(W_conv2.get_shape(), b_conv2.get_shape()))
    # produce output pool2
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

if num_pools >2: 
    with tf.name_scope('convolutional_3'):
        W_conv3 = weight_variable('W_conv3',[5, 5, features*2, features*4])
        b_conv3 = bias_variable('b_conv3',[features*4])
        print('Convolutional Layer 3 - Shape: W3:{}, b3:{}'.format(W_conv3.get_shape(), b_conv3.get_shape()))
        
        # produce output pool2
        h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
 
if num_pools >3:
    with tf.name_scope('convolutional_4'):
        W_conv4 = weight_variable('W_conv4',[5, 5, features*4, features*8])
        b_conv4 = bias_variable('b_conv4',[features*8])
        print('Convolutional Layer 4 - Shape: W4:{}, b4:{}'.format(W_conv4.get_shape(), b_conv4.get_shape()))
        
        # produce output pool2
        h_conv4 = tf.nn.relu(conv2d(h_pool3,W_conv4)+b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

# this will perform the classification
# input vector is 7x7*64: from 28 to 14 to 7 
# we'll have a vector of 7x7x64
# follow the network to find the dimmension 
with tf.name_scope('fully_connected_1'):
    W_fc1 = weight_variable('W_fc1', [int(out_height*out_width*features*pow(2,num_pools-1)), final_layer])
    b_fc1 = bias_variable('b_fc1',[final_layer])
    print('Fully Connected Layer 1 - Shape: W_fc1:{}, b_fc1:{}'.format(W_fc1.get_shape(), b_fc1.get_shape()))
    print('Feature size: {}x{}'.format(out_height,out_width))
    if num_pools == 2:
        h_pool2_flat = tf.reshape(h_pool2, [-1, int(out_height*out_width*features*pow(2,num_pools-1))])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    elif num_pools == 3:
        h_pool3_flat = tf.reshape(h_pool3, [-1, int(out_height*out_width*features*pow(2,num_pools-1))])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    elif num_pools == 4:
        h_pool4_flat = tf.reshape(h_pool4, [-1, int(out_height*out_width*features*pow(2,num_pools-1))])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    else:
        print('Wrong number of convolutional layers.')
        sys.exit()


with tf.name_scope('dropout_1'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fully_connected_2'):
    W_fc2 = weight_variable('W_fc2',[final_layer,n_classes])
    # bias vector
    b_fc2 = bias_variable('b_fc2',[n_classes])   
    print('Fully Connected Layer 2 - Shape: W_fc2:{}, b_fc2:{}'.format(W_fc2.get_shape(), b_fc2.get_shape()))
    
    y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2


loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y_conv))

# instead gradiant descent, using the Adam optimiser with a learning rate
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)


correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.InteractiveSession()

# tensorboard: useful to visualise information of the training on the network.
# watch some variables as they evolve
# specify what parameter to track: loss and the first feature map of the convolutional model.
# [0,0,0,0] position from where I want to take the slice.
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    
    if num_pools == 2:
        feat1 = tf.slice(h_conv2,[0,0,0,0],[-1,-1,-1,1])
    elif num_pools == 3:
        feat1 = tf.slice(h_conv3,[0,0,0,0],[-1,-1,-1,1])
    elif num_pools == 4:
        feat1 = tf.slice(h_conv4,[0,0,0,0],[-1,-1,-1,1])
    else:
        print('Wrong number of convolutional layers.')
        sys.exit()
    tf.summary.image('features', feat1)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(train_path, sess.graph)
# summary: notes in the computational graph, but don't affect the computation. We have to initialise them.
# add summary node for loss


# same way to compute accuracy


# where to save the monitoring data.
# to show the graph: tensorboard --logdir <file> 
# graph is available in localhost:6006, tensorflow website (tensorboard)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
#saver = tf.train.import_meta_graph('/Users/user/model.ckpt-1299.meta')
#saver.restore(sess,tf.train.latest_checkpoint('/Users/user/'))

#print('Restored session: ' +sess.run('W_conv1'))


# perform the training from here.
batch_l=0
batch_h=batch_size
print ("Starting training")
for i in range(iterations):
    print(".", end=''),
    #print("."),
    batch_i = X_train[batch_l:batch_h]
    batch_la = Y_train[batch_l:batch_h]
    batch_imgs, batch_labels = dataset6.load_images(batch_i, batch_la, resize_img_height, resize_img_width, crop_h, crop_w)
    #batch_i = train_data[0][batch_l:batch_h]
    #batch_la = train_data[0][batch_l:batch_h]
    # define values of the placeholders
    feed_dict = {x: batch_imgs, t: batch_labels, keep_prob: keep_probab}
    # every 10 iterations, run the summary and add the new entry on the output file.
    if i % 10 == 0:
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, i)
        # write the data to the disk with flush
        summary_writer.flush()
        #Check a few images
        
    
    # for each 100 iterations, print the accuracy    
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print()
        print('step %d, training accuracy %g' % (i, train_accuracy))
    # evaluate the graph of the training
    train_step.run(feed_dict=feed_dict)
    batch_l+=batch_size
    
    if batch_l < train_size:
        batch_h=batch_l+min(train_size-batch_l,batch_size)
    else:
        batch_l=0
        batch_h=batch_size

print()
print ('End time: ' + str(datetime.now().time()))
# accuracy on the test set
print()
print("Number of validation samples: " + str(len(X_test)))

batch_imgs, batch_labels = dataset6.load_images(X_test, Y_test, resize_img_height, resize_img_width, crop_h, crop_w)

print('Test accuracy %g' % accuracy.eval(feed_dict={x: batch_imgs, t: batch_labels, keep_prob: 1.0}))
mytest = sess.run(tf.argmax(y_conv,1), feed_dict={x: batch_imgs, t: batch_labels, keep_prob: 1.0})

val_count = 0
val_pointer=0
y_true = []
y_pred = []
while (val_pointer < len(X_test)):
    y_true.append(classes[batch_labels[val_pointer].argmax()])
    y_pred.append(classes[mytest[val_pointer]])
    if val_count < val_samples:
        if mytest[val_pointer] != batch_labels[val_pointer].argmax():              
            myim = batch_imgs[val_pointer] 
            io.imshow(myim)
            io.show()
            print("Predicted: " + classes[mytest[val_pointer]] + " - Real: " + classes[batch_labels[val_pointer].argmax()])  
            val_count= val_count + 1
    val_pointer = val_pointer + 1

print(confusion_matrix(y_true, y_pred, labels=classes))
    
#take checkpoints of your weights. 
# save the values of the parameters in a file
checkpoint_file = os.path.join('/Users/JoseMa/', 'model.ckpt')

saver.save(sess, checkpoint_file, global_step=iterations)
sess.close