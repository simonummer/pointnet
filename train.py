import tensorflow as tf
import numpy as np
import config as cf
import PointNet.PointNet as pn
import tools.get_data

def train_epoch(sess, op, writer):
    train_file_name = open(cf.DIR_MODELNET40+"train_files.txt")
    train_file_n = [line.rstrip() for line in train_file_name]
    train_file_name.close()
    train_file_index = np.arange(0, len(train_file_n))
    np.random.shuffle(train_file_index)
    
    for index in range(len(train_file_index)):
        print('********LOADING_FILE:', index, '***********')
        points_data, label = tools.get_data.get_data(cf.DIR_MODELNET40 + train_file_n[train_file_index[index]] )
        
        #shuffle
        idx = np.arange(len(label))
        points_data = points_data[idx, 0: cf.sample_point_num, :]#assume it is shuffled in every object  TODOTODOTODO
        label = label[idx]
        
        temp = np.zeros((len(label), 40))
        for i in range(len(label)):
            temp[i][label[i]] = 1
        label = temp
        
        batch_max = len(label) // cf.batch_size
        
        for batch_idx in range(batch_max):
            start = batch_idx * cf.batch_size
            end = (batch_idx + 1) * cf.batch_size 
            
            inputs = points_data[start : end , ...]
            truth = label[start : end, ...]
            
            feed_dict = {points: inputs, ground_truth: truth}
            
            [a,l,u,sum,step] = sess.run(op, feed_dict = feed_dict)

            writer.add_summary(sum, step)
            print('batch:',batch_idx + 1,' train_accuracy = ',a,' train_loss = ',l, 'step',step)
            
def eval_epoch(sess, op, writer):
    test_file_name = open(cf.DIR_MODELNET40 + 'test_files.txt')
    test_file = [line.rstrip() for line in test_file_name]
    test_file_name.close()
    tot_acc = 0
    tot_seen = 0
    tot_los = 0

    for index in range(len(test_file)):
        pointss, label = tools.get_data.get_data(cf.DIR_MODELNET40+test_file[index])
        pointss = pointss[:, 0:cf.sample_point_num, :]

        temp = np.zeros((len(label),40))
        for i in range(len(label)):
            temp[i][label[i]] = 1
        label = temp

        for batch_idx in range(len(label)//cf.batch_size):
            start_idx = batch_idx*cf.batch_size
            end_idx = (batch_idx+1) * cf.batch_size

            inputs = pointss[start_idx:end_idx, ...]
            truth = label[start_idx:end_idx, ...]
            feed_dic = {points: inputs, ground_truth: truth}

            [acc, los] = sess.run(op, feed_dict = feed_dic)

            tot_acc += acc
            tot_seen += 1
            tot_los += los

    acc = tot_acc / tot_seen
    los = tot_los / tot_seen

    print("%%%%%%%%%%%%%%%%%%%%%%%%  evaluate  %%%%%%%%%%%%%%%%%%%%%%%%")
    print('test avg accuracy :', acc)
    print('test avg loss :', los)




def get_learning_rate(batch):
    rate = tf.train.exponential_decay(0.0001, batch * cf.batch_size, 200000, 0.7, staircase=True)
    rate = tf.maximum(rate, 0.000005)
    return rate
        
    
#print(test.get_shape())
with tf.Graph().as_default():
    with tf.device('/gpu:0'):
        #
        points = tf.placeholder(tf.float32, (cf.batch_size, cf.sample_point_num, 3))
        ground_truth = tf.placeholder(tf.int32, (cf.batch_size, 40))
        pred, matr= pn.get_net(points)
        loss = pn.get_loss(pred, ground_truth, matr)
        tf.summary.scalar('loss', loss)
        #test loss
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.to_int64(ground_truth),1))
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(cf.batch_size)
        tf.summary.scalar('accuracy', accuracy)

        batch = tf.Variable(0)
        Rate = get_learning_rate(batch)
        tf.summary.scalar('learning_rate', Rate)
        
        optimizer = tf.train.AdamOptimizer(Rate)
        train_op = optimizer.minimize(loss,global_step=batch)
        
    #sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True,gpu_options = gpu_options))
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(cf.MAX_EPOCH):
        print('********EPOCH:', epoch, '***********')

        train_epoch(sess, [accuracy, loss, train_op, merged, batch], train_writer)
        eval_epoch(sess, [accuracy, loss], test_writer)
        
        if epoch % 10 == 0 :
            save_path = saver.save(sess = sess, save_path = cf.DIR_MODELNET40+'model.ckpt')
            print("save")
                    

