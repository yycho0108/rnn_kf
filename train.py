#!/usr/bin/python
import numpy as np

import tensorflow as tf
import random

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
slim = tf.contrib.slim

from generate_data import GenerateData_v2 as gen

## ==========
##   MODEL
## ==========
#
# Parameters
learning_rate = 0.01
training_iters = 5000
batch_size = 128
display_step = 10

# Network Parameters
max_seq_len = 200 # Sequence max length
min_seq_len = 50 # Sequence max length

n_hidden = 64 # hidden layer num of features

FINAL_TENSOR_NAME='prediction'

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

def net(x,l):

    cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(n_hidden),
            input_keep_prob = 0.5) # num memory units in cell
    #outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32, sequence_length=1)
    #states = [batch_size, max_seq, channels]

    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32, time_major=False, sequence_length=l)

    # outputs = [batch_size, max_time, cell.output_size]

    batch_range = tf.range(tf.shape(outputs)[0])
    indices = tf.stack([batch_range,l-1], axis=1)

    o = tf.gather_nd(outputs,indices)

    #w_= tf.truncated_normal([n_hidden, 2], stddev=0.01)
    #w = tf.Variable(w_, name='W')
    #variable_summaries(w,'W')
    #b = tf.Variable(tf.zeros([2]), name='b')
    #variable_summaries(b,'b')
    #o = tf.matmul(outputs,w) + b
    #o = tf.tanh(o)
    #o = tf.nn.relu(o)

    #o = slim.fully_connected(o, n_hidden, activation_fn=tf.nn.relu, scope='fc/fc1')
    o = slim.fully_connected(o, 2, activation_fn=tf.tanh, scope='fc/fc2')

    #h = slim.fully_connected(outputs, n_hidden, scope='fc/fc1')
    #o = slim.fully_connected(h, n_hidden, scope='fc/fc2')
    #o = slim.fully_connected(o, 1, activation_fn=tf.tanh, scope='fc/fc3')

    vs = slim.variables.trainable_variables()
    for v in vs:
        variable_summaries(v, 'slim')
        
    o = tf.identity(o, name=FINAL_TENSOR_NAME)

    return o

#def format_data(data, n_input):
#    res = []
#    for i in range(len(data) - n_input):
#        res.append(data[:, i:i+n_input, :])
#    res = np.array(res)
#    #return res #[?, batch_size, n_input, channels]
#    return np.swapaxes(res, 0, 1) #[batch_size, split_samples, n_input, channels]

def main():

    graph = tf.Graph()
    with graph.as_default():

        x = tf.placeholder(tf.float32, [None, max_seq_len, 2], name='x') # input : position with noise
        y = tf.placeholder(tf.float32, [None, 2], name='y') # label : ground truth position
        l = tf.placeholder(tf.int32, [None], name='l') # seq. length

        pred = net(x,l)
        cost = tf.nn.l2_loss(y - pred)
        tf.summary.scalar('cost', cost)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        init = tf.global_variables_initializer()

    sess = tf.Session()

    g = gen(max_seq_len = max_seq_len, min_seq_len = max_seq_len, noise=5e-2)

    with tf.Session(graph=graph) as sess:
        sess.run(init)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/localize_logs', sess.graph)

        for step, label, data, len in g.get(batch_size, training_iters):
            feed_dict = {x : data, y : label, l : len}
            # label = [batch, length, channel]
            summary, _ = sess.run([merged, opt], feed_dict=feed_dict)
            writer.add_summary(summary, step)
            if step % 100 == 0:
                prediction, loss = sess.run([pred, cost], feed_dict=feed_dict)
                print('[Step %02d] loss : %f]' % (step, loss))
                print('real :' , label[0])
                print('pred :' , prediction[0])

        output_graph_def = graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
        with gfile.FastGFile('output_graph.pb', 'wb') as f:
          f.write(output_graph_def.SerializeToString())

if __name__ == "__main__":
    main()

#pred = dynamicRNN(x, seqlen, weights, biases)
#
## Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#
## Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
## Initializing the variables
#init = tf.global_variables_initializer()
#
## Launch the graph
#with tf.Session() as sess:
#    sess.run(init)
#    step = 1
#    # Keep training until reach max iterations
#    while step * batch_size < training_iters:
#        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
#        # Run optimization op (backprop)
#        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                       seqlen: batch_seqlen})
#        if step % display_step == 0:
#            # Calculate batch accuracy
#            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,
#                                                seqlen: batch_seqlen})
#            # Calculate batch loss
#            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,
#                                             seqlen: batch_seqlen})
#            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.5f}".format(acc))
#        step += 1
#    print("Optimization Finished!")
#
#    # Calculate accuracy
#    test_data = testset.data
#    test_label = testset.labels
#    test_seqlen = testset.seqlen
#    print("Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label,
#                                      seqlen: test_seqlen}))
