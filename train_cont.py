#!/usr/bin/python
import numpy as np

import tensorflow as tf
import random

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
slim = tf.contrib.slim

from generate_data import GenerateData_v2_2 as gen

import cv2

import os

## ==========
##   MODEL
## ==========
#
# Parameters
learning_rate = 0.005
training_iters = 500
testing_iters = 10
batch_size = 128
display_step = 10

load_ckpt = True 
log_root = '/tmp/localize_logs/'
ckpt_path='data/model.ckpt'

do_train = False 
do_test = True

# Network Parameters
seq_len = 200 # Observation Sequence length

n_hidden = 64 # hidden layer num of features

MODEL_SCOPE = 'model'

def variable_summaries(var, scope='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def net(x, batch_size, reuse=None):
    #i = tf.placeholder(tf.float32, [2,None,n_hidden], name='i')
    #i_t = tf.contrib.rnn.LSTMStateTuple(i[0], i[1])

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, reuse=reuse) # num memory units in cell
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=1.0)
    cell = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)

    i_s = tf.Variable(cell.zero_state(batch_size, tf.float32), trainable=False) # this one is special!
    i_t = tuple(tf.unstack(i_s, axis=0))
    i_t = tuple(tf.contrib.rnn.LSTMStateTuple(t[0], t[1]) for t in i_t)
    #print i_t[0].shape
    #i_t = () + (tf.contrib.rnn.LSTMStateTuple(i_s[0][0], i_s[0][1]),)

    #outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32, sequence_length=1)
    #states = [batch_size, max_seq, channels]

    outputs, states = tf.nn.dynamic_rnn(cell, x, initial_state = i_t, dtype=tf.float32, time_major=False)

    # outputs = [batch_size, max_time, cell.output_size]

    batch_range = tf.range(tf.shape(outputs)[0])

    o = slim.fully_connected(outputs, 2, activation_fn=tf.tanh, weights_regularizer=slim.l2_regularizer(0.01), scope='fc')
    
    vs = slim.variables.trainable_variables()
    for v in vs:
        variable_summaries(v, 'slim')

    o = tf.identity(o, name='pred')

    reset = tf.assign(i_s, tf.zeros_like(i_s), name='reset')
    update = tf.assign(i_s, states, name='update')

    return o, reset, update

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def main():

    graph = tf.Graph()
    with graph.as_default():

        x = tf.placeholder(tf.float32, [None, 1, 2], name='x') # input : position with noise
        y = tf.placeholder(tf.float32, [None, 1, 2], name='y') # label : ground truth position

        with tf.variable_scope(MODEL_SCOPE, reuse=None):
            train = {}
            train['pred'], train['reset'], train['update'] = net(x, batch_size, None)

        with tf.variable_scope(MODEL_SCOPE, reuse=True):
            test = {}
            test['pred'], test['reset'], test['update'] = net(x, 1, True)

        cost = tf.nn.l2_loss(y - train['pred'])
        tf.summary.scalar('cost', cost)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        init = tf.global_variables_initializer()
        for v in tf.trainable_variables():
            print v.name, v.shape

    sess = tf.Session()

    g = gen(seq_len = seq_len, noise=4e-2)

    with tf.Session(graph=graph) as sess:
        saver = tf.train.Saver()

        sess.run(init)

        if load_ckpt:
            saver.restore(sess, ckpt_path)

        ##  TRAIN  ##
        if do_train:

            ## LOGGING ##
            if not os.path.exists(log_root):
                os.makedirs(log_root)
            run_id = len(os.walk('/tmp/localize_logs/').next()[1])
            writer = tf.summary.FileWriter(os.path.join('/tmp/localize_logs/', ('run_%02d' % run_id)) , graph)
            merged = tf.summary.merge_all()
            #############

            for step, label, data in g.get(batch_size, training_iters):
                _ = sess.run(train['reset'])
                for x_in, y_in in zip(data, label):
                    feed_dict = {x : x_in[:,np.newaxis,:], y : y_in[:,np.newaxis,:]}
                    # label = [batch, length, channel]
                    summary, _, _= sess.run([merged, opt, train['update']], feed_dict=feed_dict)
                    writer.add_summary(summary, step)

                if step % 100 == 0:
                    prediction, loss = sess.run([train['pred'], cost], feed_dict=feed_dict)
                    print('[Step %02d] loss : %f]' % (step, loss))
                    #print('real :' , label[-1])
                    #print('pred :' , prediction[-1])
            saver.save(sess, ckpt_path)
        ############

        ##  SAVE  ##
        output_graph_def = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), ['model_1/pred', 'model_1/update', 'model_1/reset'])

        with gfile.FastGFile('output_graph.pb', 'wb') as f:
          f.write(output_graph_def.SerializeToString())
        ############

        ## TESTING ##
        if do_test:
            w,h = 512,512
            frame = np.zeros((h,w,3), dtype=np.uint8)

            for step, label, data in g.get(1, testing_iters):
                frame.fill(0)
                _ = sess.run(test['reset'])

                prvx = None
                prvy = None
                prvp = None
                err_m = 0
                err_p = 0

                for x_in, y_in in zip(data, label):
                    feed_dict = {x : x_in[:, np.newaxis,:], y : y_in[:, np.newaxis, :]}
                    pred, _ = sess.run([test['pred'], test['update']], feed_dict=feed_dict)

                    x_in, y_in, pred = [(np.squeeze(e) * [h/2,w/2] + [h/2,w/2]).astype(np.int32) for e in (x_in, y_in, pred)]

                    y_pt = (y_in[1], y_in[0])
                    x_pt = (x_in[1], x_in[0])
                    p_pt = (pred[1], pred[0])

                    err_m += np.linalg.norm(y_in - x_in)
                    err_p += np.linalg.norm(y_in - pred)

                    cv2.circle(frame, y_pt, 5, (255,0,0), thickness=-1) # --> true pos, blue
                    cv2.circle(frame, x_pt, 4, (0,255,0), thickness= 1) # --> measured pos, green
                    cv2.circle(frame, p_pt, 4, (0,0,255), thickness= 1) # --> predicted pos, reg

                    if prvx is not None:
                        cv2.line(frame, prvy, y_pt, (255,0,0), 1)
                        cv2.line(frame, prvx, x_pt, (0,255,0), 1)
                        cv2.line(frame, prvp, p_pt, (0,0,255), 1)

                    prvy = y_pt
                    prvx = x_pt
                    prvp = p_pt

                    cv2.imshow('frame', frame)
                    if cv2.waitKey(20) == 27:
                        break
                print('err_m : %.2f ; err_p : %.2f' % (err_m, err_p))
                if cv2.waitKey(0) == 27:
                    break
        #############



if __name__ == "__main__":
    main()
