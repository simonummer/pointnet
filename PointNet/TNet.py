import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cf
import numpy as np

#默认K=3
    
def get_input_TNet( point_cloud ,bn_decay = 0.999):
    b_size = cf.batch_size
    n_size = cf.sample_point_num
    
    inputs = tf.expand_dims(point_cloud, -1)#channel维
    
    #print(inputs.get_shape())

    net = slim.conv2d(inputs, 64, [1,3],1,'VALID',scope='tconv1')
    net = slim.batch_norm(net, decay = bn_decay, scope='tbn1')
    net = slim.conv2d(net, 128, [1,1],1,'VALID',scope='tconv2')
    net = slim.batch_norm(net, decay = bn_decay, scope='tbn2')
    net = slim.conv2d(net, 1024, [1,1],1,'VALID',scope='tconv3')
    net = slim.batch_norm(net, decay = bn_decay, scope='tbn3')
    net = slim.max_pool2d(net, [n_size,1], scope = 'tpool_1')
    
    net = tf.reshape(net, [b_size, -1])

    net = slim.fully_connected(net, 512 ,scope='tfc1')
    net = slim.batch_norm(net, decay = bn_decay, scope='tbn4')
    net = slim.fully_connected(net, 256 ,scope='tfc2')
    net = slim.batch_norm(net, decay = bn_decay, scope='tbn5')
    
    with tf.variable_scope('transform_net'):
        
        weights = slim.variable('weights', 
                                shape = [256, 3*3], 
                                initializer = tf.constant_initializer(0.0), 
                                dtype = tf.float32)
        
        biases = slim.variable('biases',
                               shape = [3*3],
                               initializer = tf.constant_initializer(0.0),
                               dtype = tf.float32)
        
        biases += tf.constant([1,0,0,0,1,0,0,0,1],dtype = tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)  
        
    transform = tf.reshape(transform, [b_size, 3, 3])
    return transform

#K=64
def get_feature_TNet( point_cloud,bn_decay = 0.999):
    b_size = cf.batch_size
    n_size = cf.sample_point_num


    net = slim.conv2d(point_cloud,64, [1,1], 1,'VALID', scope = 'ftconv1')
    net = slim.batch_norm(net,decay=bn_decay,scope='ftbn1')
    net = slim.conv2d(net,128, [1,1], 1, 'VALID', scope = 'ftconv2')
    net = slim.batch_norm(net,decay=bn_decay,scope='ftbn2')
    net = slim.conv2d(net,1024,[1,1], 1, 'VALID', scope = 'ftconv3')
    net = slim.batch_norm(net,decay=bn_decay,scope='ftbn3')
    net = slim.max_pool2d(net, [n_size,1], scope = 'ftpool_1')
    
    net = tf.reshape(net, [b_size, -1])

    net = slim.fully_connected(net,512,scope='fttc1')
    net = slim.batch_norm(net,decay=bn_decay,scope='ftbn4')
    net = slim.fully_connected(net,256,scope='fttc2')
    net = slim.batch_norm(net,decay=bn_decay,scope='ftbn5')
    
    with tf.variable_scope('transform_feat_net'):
        
        weights = slim.variable('weights', 
                                shape = [256, 64*64], 
                                initializer = tf.constant_initializer(0.0), 
                                dtype = tf.float32)
        
        biases = slim.variable('biases',
                               shape = [64*64],
                               initializer = tf.constant_initializer(0.0) ,
                               dtype = tf.float32)
        
        biases += tf.constant(np.eye(64).flatten(), dtype=tf.float32)#代码这么写的 
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
        
    transform = tf.reshape(transform, [b_size, 64, 64])
    return transform
