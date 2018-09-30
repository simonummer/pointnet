import tensorflow as tf

saver = tf.train.import_meta_graph('/home/simonummer/下载/modelnet40_ply_hdf5_2048/model.ckpt.meta')
sess = tf.Session()

saver.restore(sess, tf.train.latest_checkpoint)
