import cv2
import numpy as np 
import time
from threading import Thread


class CalibrationScreen():

    def __init__(self, width, height, lines, columns, img, pipe, planes=1):
        self.w = width
        self.h = height
        self.L = lines
        self.C = columns
        self.target = img
        self.border = 24
        self.pipe = pipe
        self.planes = planes
        self.run()

    
    def run(self):
        size = (self.h - self.border)//4
        self.target = cv2.resize(self.target, (size,size))
        hgap = ((self.w - self.border)-(4*size))//5
        vgap = ((self.h - self.border)-(3*size))//5
        for p in range(len(self.planes)):
            self.__cycle(hgap, vgap, size)


    def __cycle(self, hgap, vgap, size):
        v = vgap + self.border//2
        screen = np.ones((self.h, self.w), np.uint8)*255
        self.pipe.send(screen)
        msg = self.pipe.recv()
        for i in range(self.L):
            h = hgap + self.border//2
            for j in range(self.C):
                screen = np.ones((self.h, self.w), np.uint8)*255
                screen[v:v+size,h:h+size] = self.target
                h += hgap + size
                self.pipe.send(screen)
                if self.pipe.poll(2.5):
                    msg = self.pipe.recv()
                    continue
            v += vgap + size
        