#!/usr/bin/python

import numpy as np
import cv2
import os
import tensorflow as tf

class GenerateData(object):
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
            self.est.append(5e-2 * np.random.randn(2) + pos)

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
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

    create_graph()
    with tf.Session() as sess:
        report_graph(sess.graph)
        pred = sess.graph.get_tensor_by_name('prediction:0')


        def run(frame, pos, est):
            l = np.random.randint(199,200)

            h,w,_ = frame.shape

            if len(pos) > l:
                x = np.reshape(np.array(est[-l:]),[1,l,2])
                y = sess.run(pred,feed_dict={
                    'x:0' : x,
                    'l:0' : np.array([l])
                    })[0]

                draw_pt(frame, y, w, h, 3, (255,0,0), -1)

            if len(pos) > 1:
                draw_pt(frame, est[-1], w, h, 5, (0,255,0), -1)
                draw_pt(frame, pos[-1], w, h, 3, (0,0,255), -1)

        gen = GenerateData(shape=(480,640,3))
        gen.start(run)

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
