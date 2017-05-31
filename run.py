#!/usr/bin/python

import numpy as np
import cv2
import os
import tensorflow as tf

from train_cont import net
class GameRun(object):
    def __init__(self, name='', shape=(480,640)):
        if len(name) <= 0:
            name = 'frame'

        self.name = name
        self.shape = shape
        self.frame = np.zeros(shape=shape)
        self.pos = []
        self.est = []

    def start(self, callback):

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.track)
        cv2.imshow(self.name, self.frame)

        while True:
            cv2.imshow(self.name, self.frame)
            k = cv2.waitKey(20)

            if len(self.pos) > 0:
                self.est.append(5e-2 * np.random.randn(2) + self.pos[-1])

            callback(self.frame, self.pos, self.est)

            if k == 27 or k == ord('q'):
                break

    def track(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            h,w,_ = self.shape
            h,w = float(h), float(w)
            n_x = float(x) / (w/2) - 1 # -1 ~ 1
            n_y = float(y) / (h/2) - 1 # -1 ~ 1
            pos = np.array([n_x,n_y])
            self.pos.append(pos)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print 'op', node.op
        _ = tf.import_graph_def(graph_def, name='')


def report_graph(graph):
  for op in graph.get_operations():
     print('===')
     print(op.name)
     print('Input:')
     for i in op.inputs:
         print('\t %s' % i.name, i.get_shape())
     print('Output:')
     for o in op.outputs:
         print('\t %s' % o.name, o.get_shape())
     print('===')

def draw_pt(frame, pt, w, h, sz, col, t):
    pt = (pt + 1) * np.array([w/2.,h/2.])
    pt = tuple((int(e) for e in pt))
    cv2.circle(frame, pt, sz, col, thickness=t)

def main():
    global dm,de

    create_graph()

    with tf.Session() as sess:
        
        report_graph(sess.graph)

        #with tf.variable_scope('model', reuse=True):
        #    test = {}
        #    test['pred'], test['reset'], test['update'] = net(x, 1, True)

        x = sess.graph.get_tensor_by_name('x:0')
        pred = sess.graph.get_tensor_by_name('model_1/pred:0')
        #print pred.shape

        dm = 0.0
        de = 0.0

        def run(frame, pos, est):
            global dm,de

            h,w,_ = frame.shape

            if len(est) > 128:
                x = np.reshape(np.array(est),[1,-1,2])
                y = sess.run(pred,feed_dict={
                    'x:0' : x,
                    })[0]

                draw_pt(frame, y, w, h, 3, (255,0,0), -1)

                dm += np.linalg.norm(pos[-1] - est[-1])
                de += np.linalg.norm(pos[-1] - y)
                print 'dm,de', dm,de


            if len(pos) > 1:
                draw_pt(frame, est[-1], w, h, 5, (0,255,0), -1)
                draw_pt(frame, pos[-1], w, h, 3, (0,0,255), -1)

        game = GameRun(shape=(480,640,3))
        game.start(run)

if __name__ == "__main__":
    main()

    #while True:
    #    cv2.circle(frame, tuple(int(e) for e in d_p), 3, 128)
    #    cv2.circle(frame, tuple(int(e) for e in g_p), 3, 255)
    #    cv2.imshow('frame', frame)
    #    cv2.waitKey(10)

    #gen = GenerateData()
    #gen.start()
    #gen.save()
