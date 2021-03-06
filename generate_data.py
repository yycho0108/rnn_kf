#!/usr/bin/python

import numpy as np
import cv2
import os
from perlin import PerlinNoise

class GenerateData(object):
    def __init__(self, name='', shape=(480,640)):
        if len(name) <= 0:
            name = 'frame'

        self.name = name
        self.shape = shape
        self.frame = np.zeros(shape=shape)
        self.pos = []

    def start(self):

        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, self.track)
        cv2.imshow(self.name, self.frame)

        while True:
            cv2.imshow(self.name, self.frame)
            k = cv2.waitKey(20)
            if k == 27 or k == ord('q'):
                break

    def track(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            h,w = self.shape
            h,w = float(h), float(w)
            self.pos.append([x/w,y/h]) # normalize
            cv2.circle(self.frame, (x,y), 3, 255)

    def save(self):
        idx = 0
        for sub in os.listdir('data'):
            if sub.endswith('.npy'):
                idx += 1
        name = ('seq_%03d.npy' % idx)
        name = os.path.join('data', name)
        print ('data length : %d' % len(self.pos))
        np.save(name, np.asarray(self.pos), allow_pickle=True)

class GenerateData_v2(object):
    def __init__(self, max_seq_len=20, min_seq_len=3, noise=5e-2):

        self.max_seq_len = max_seq_len 
        self.min_seq_len = min_seq_len
        self.noise = noise

    def get(self, batch_size=1, num_steps=1, final_label=True):
        for i in range(num_steps):
            n = self.max_seq_len
            acc = np.random.randn(batch_size, 2 * n, 2) + np.random.randn(batch_size, 1, 2) #2 = x,y
            #acc -= np.mean(acc, axis=1)[:, np.newaxis, :]
            vel = np.cumsum(acc, axis=1) + np.random.randn(batch_size, 2 * n, 2)
            #vel[np.abs(vel) > 5] *= 0.2


            pos = np.cumsum(vel, axis=1) + self.noise * np.random.randn(batch_size, 2 * n, 2)
            mx = np.max(np.abs(pos), axis=1)

            #pos += (mx * np.random.rand(batch_size, 2))[:, np.newaxis, :] # initial offset

            mx = np.max(np.abs(pos), axis=1)
            pos /= mx[:,np.newaxis,:]

            len = np.random.randint(self.min_seq_len, self.max_seq_len+1, size=batch_size)

            data = pos[:,n/2 : n+n/2,:] + self.noise * np.random.randn(batch_size, n, 2)

            if final_label:
                #label = np.take(pos, n/2+len, axis=1)
                label = pos[range(batch_size), n/2+len, :]
            else:
                label = pos[range(batch_size), n/2:n/2+len,:]

            yield i, label, data, len

class GenerateData_v2_2(object):
    def __init__(self, seq_len=10, noise=5e-2):
        self._seq_len = seq_len
        self._noise = noise

    def get(self, batch_size=1, num_steps=1):
        for i in range(num_steps):

            n = self._seq_len * 2

            a = 0.02 * np.random.normal(size=(batch_size, 2*n))
            t_init = (2 * np.pi * np.random.random())
            p_init = np.random.uniform(size=2)
            t = t_init + np.cumsum(np.cumsum(a, axis=1), axis=1)

            v = 0.2 * np.random.uniform(size=(batch_size, 2*n))
            vel = np.stack([v * np.cos(t), v * np.sin(t)], axis=-1)

            pos = p_init + np.cumsum(vel, axis=1)
            
            mx = np.max(np.abs(pos), axis=1)
            pos /= mx[:,np.newaxis,:]

            data = pos[:,n/2 : n/2+self._seq_len,:] + self._noise * np.random.randn(batch_size, self._seq_len, 2)

            label = pos[range(batch_size), n/2:n/2+self._seq_len, :]

            yield i, np.transpose(label,(1,0,2)), np.transpose(data,(1,0,2))

class GenerateData_v3(GenerateData_v2):
    def __init__(self, max_seq_len=20, min_seq_len=3, noise=5e-2, frame_size=(480,640), ball_size=20):
        super(GenerateData_v3, self).__init__(max_seq_len, min_seq_len, noise)
        self._frame_size = frame_size
        self._ball_size = ball_size
    def ball_frame(self, pos):
        frame = np.random.random_sample(size = self._frame_size)
        #frame = np.zeros(np.float32, shape = self._frame_size)
        cv2.circle(frame, tuple(int(e) for e in pos), self._ball_size, 1.0, thickness=-1)
        return frame
    def get(self, batch_size=1, num_steps=1, final_label=True):

        h,w = self._frame_size
        p = PerlinNoise((self.max_seq_len,h,w))

        for (i, label, data, len) in GenerateData_v2.get(self, batch_size, num_steps, final_label):
            # data = [batch_size, seq_len, pos]
            b, l, _ = data.shape

            # data_r = data with images
            data_r = np.stack([p.getData(scale=32.0, reset=True) for _ in range(batch_size)], axis=0)
            label_r = np.zeros((b, h, w), dtype=np.float32)

            for idx in np.ndindex(b,l):
                pos = tuple(int(e) for e in [w/2. * (data[idx][0]+1) , h/2. * (data[idx][1]+1)])
                cv2.circle(data_r[idx], pos, self._ball_size, 0.5, thickness=-1)

            for idx in range(b):
                pos = tuple(int(e) for e in [w/2. * (label[idx][0]+1) , h/2. * (label[idx][1]+1)])
                cv2.circle(label_r[idx], pos, self._ball_size, 1.0, thickness=-1)

            yield i, label_r, data_r, len


def main():
    gen = GenerateData_v2(200,100,5e-2)
    w = 640
    h = 480
    c = [w/2.,h/2.]
    frame = np.zeros((h,w), dtype=np.uint8)

    # iterate over batch
    for i, g_pos_b, d_pos_b, len_b in gen.get(batch_size=3, num_steps = 1):
        for g_pos, d_pos, len in zip(g_pos_b, d_pos_b,len_b):

            g_p = g_pos * np.array([w/2.,h/2.]) + c
            cv2.circle(frame, tuple(int(e) for e in g_p), 3, 255, thickness=-1)

            for d_p in d_pos[:len]:
                d_p = d_p * np.array([w/2.,h/2.]) + c
                cv2.circle(frame, tuple(int(e) for e in d_p), 3, 128)
                cv2.imshow('frame', frame)
                k = cv2.waitKey(20)

                if k == 27 or k == ord('q'):
                    return

            cv2.waitKey(0)

def main_v2_2():
    gen = GenerateData_v2_2(seq_len=100)

    w = 640
    h = 480
    c = [w/2.,h/2.]
    frame = np.zeros((h,w), dtype=np.uint8)

    #for step, label, data in g.get(batch_size=3, num_steps=1):
    # iterate over batch
    for i, g_pos_b, d_pos_b in gen.get(batch_size=1, num_steps = 1):
        g_pos_b = np.squeeze(g_pos_b)
        d_pos_b = np.squeeze(d_pos_b)

        for g_pos, d_pos in zip(g_pos_b, d_pos_b):

            g_p = g_pos * np.array([w/2.,h/2.]) + c
            cv2.circle(frame, tuple(int(e) for e in g_p), 3, 255, thickness=-1)

            d_p = d_pos * np.array([w/2.,h/2.]) + c
            cv2.circle(frame, tuple(int(e) for e in d_p), 3, 128)

            cv2.imshow('frame', frame)
            k = cv2.waitKey(20)

            if k == 27 or k == ord('q'):
                return


def main_v3():
    gen = GenerateData_v3(200,100,5e-2, frame_size=(60,80), ball_size=5)
    for i, lab_b, dat_b, len_b in gen.get(batch_size=3, num_steps = 1):
        for lab, dat_s, len in zip(lab_b, dat_b, len_b): # seq_len ...
            for dat in dat_s[:len]:
                cv2.imshow('frame', dat)
                cv2.imshow('label', lab)
                k = cv2.waitKey(20)
                if k == 27 or k == ord('q'):
                    return


if __name__ == "__main__":
    main_v2_2()

    #gen = GenerateData()
    #gen.start()
    #gen.save()
