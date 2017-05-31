from numpy import array, abs, arange, dot, int8, int32, floor, fromfunction,\
                  hypot, ones, prod, random, indices, newaxis, poly1d
import numpy as np

import Image

import cv2

class PerlinNoise(object):

    def __init__(self, size=None, n=None):

        self.n = n if n else  256        
        self.size = size if size else (256, 256)

        self.order = len(self.size)
        
        self.G = 2 * np.random.random_sample(size=(self.n, self.order)) - 1 # -1 ~ 1
        self.G /= np.sqrt(self.order) # guaranteed to fall within unit sphere

        self.P = arange(self.n, dtype=int32)
        
        random.shuffle(self.P)
        
        self.idx_ar = indices(2*ones(self.order,dtype=int8), dtype=int8).reshape(self.order, -1).T
        self.drop = poly1d((-6, 15, -10, 0, 0, 1.0))

    def noise(self, coords):

        ijk = (floor(coords) + self.idx_ar).astype(int8)

        uvw = coords - ijk

        indexes = self.P[ijk[:,:, self.order - 1]]

        for i in range(self.order - 1):
            indexes = self.P[(ijk[:,:, i] + indexes) % len(self.P)]

        gradiens = self.G[indexes % len(self.G)]
#        gradiens = self.G[(ijk[:,:, 0] + indexes) % len(self.G)]
        
        res = (self.drop(abs(uvw)).prod(axis=2)*prod([gradiens, uvw], axis=0).sum(axis=2)).sum(axis=1)

        # normalize
        mn = np.min(res)
        mx = np.max(res)
        res -= mn # 0 ~ (mn+mx)
        res /= (mx-mn)

        return res

    def getData(self, scale=32.0, reset=False):
        if reset:
            self.G = 2 * np.random.random_sample(size=(self.n, self.order)) - 1 # -1 ~ 1
            self.G /= np.sqrt(self.order) # guaranteed to fall within unit sphere
        return self.noise(indices(self.size).reshape(self.order, 1, -1).T / scale).reshape(self.size)

    def saveImage(self, fileName, scale=32.0):
        im = self.getImage(scale)
        im.save(fileName)


if __name__ == "__main__":
    # 1D Noise

    shape = (240,320)
    n = PerlinNoise(size=shape)
    for _ in range(5):
        p = n.getData(scale=64.0, reset=True)
        cv2.imshow('p', p)
        cv2.waitKey(0)

    #d = n.getData(scale=64.0).reshape(shape)
    #for g in np.split(d,seq_len,axis=-1):
    #    cv2.imshow('g', g)
    #    cv2.waitKey(0)

    #seq_len = 100
    #shape = (seq_len, 120, 160)
    #n = PerlinNoise(size=shape)
    #d = n.getData(scale=64.0)
    #for g in d:
    #    cv2.imshow('g', g)
    #    cv2.waitKey(0)



