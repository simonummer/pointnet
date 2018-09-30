import tensorflow as tf
import config as cf
import tensorflow.contrib.slim as slim
import PointNet.TNet as TNet 
import numpy as np


    
def get_net(inputs, bn_decay = 0.999):#main body
    """
    inputs: batch_size*sample_point_num*3
    
    """
    
    with tf.variable_scope("pointnet",reuse = tf.AUTO_REUSE ):
        b_size = cf.batch_size
        n_size = cf.sample_point_num
        
        transform = TNet.get_input_TNet(inputs)
        
        traned = tf.matmul(inputs,transform)
        
        net = tf.expand_dims(traned, -1)
        #print(net.get_shape())
        net = slim.conv2d(net, 64 , [1,3], 1,'VALID',scope= 'conv1' )
        net = slim.batch_norm(net, decay = bn_decay, scope = 'bn1')
        net = slim.conv2d(net, 64 , [1,1], 1,'VALID',scope= 'conv2' )
        net = slim.batch_norm(net, scope = 'bn2',decay = bn_decay)
        transform = TNet.get_feature_TNet(net)
        
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])# net = tf.matmul(net, transform)
        

        net = slim.conv2d(net, 64 , [1,1], 1,'VALID',scope= 'conv3' )
        net = slim.batch_norm(net, scope = 'bn3',decay = bn_decay)
        net = slim.conv2d(net, 128 , [1,1], 1,'VALID',scope= 'conv4' )
        net = slim.batch_norm(net, scope = 'bn4',decay = bn_decay)
        net = slim.conv2d(net, 1028 , [1,1], 1,'VALID',scope= 'conv5' )
        net = slim.batch_norm(net, scope = 'bn5',decay = bn_decay)
        net = slim.max_pool2d(net, [n_size,1], scope='pool')
        
        net = tf.reshape(net, [b_size,-1])
        net = slim.fully_connected(net, 512, scope='fc1')
        net = slim.batch_norm(net, scope = 'fc_bn1',decay = bn_decay)
        net = slim.fully_connected(net, 256, scope='fc2')
        net = slim.batch_norm(net, scope = 'fc_bn2',decay = bn_decay)
        net = slim.fully_connected(net, 40, scope='fc3',activation_fn=None)
    
    return net, transform
    
def get_loss(pred, label, tran_matrix, reg_weight=0.001):
    loss = slim.losses.softmax_cross_entropy(pred, label)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', loss)
    
    K = 64
    mat_dif = tf.matmul(tran_matrix, tf.transpose(tran_matrix, perm=[0,2,1]))
    mat_dif -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_dif_loss = tf.nn.l2_loss(mat_dif) 
    tf.summary.scalar('mat loss', mat_dif_loss)

    return loss + mat_dif_loss * reg_weight
    
        
    
    