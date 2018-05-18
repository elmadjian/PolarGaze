import cv2
import numpy as np 
import time
from threading import Thread


class CalibrationScreen(Thread):

    def __init__(self, width, height, lines, columns, img, cv):
        Thread.__init__(self)
        self.w = width
        self.h = height
        self.L = lines
        self.C = columns
        self.target = img
        self.border = 40
        self.cv = cv

    
    def run(self):
        size = (self.h - self.border)//4
        self.target = cv2.resize(self.target, (size,size))
        hgap = ((self.w - self.border)-(4*size))//5
        vgap = ((self.h - self.border)-(3*size))//5
        v = vgap + self.border//2
        screen = np.ones((self.h, self.w), np.uint8)*255
        window = cv2.namedWindow('calibration screen')
        cv2.imshow(window, screen)
        k = cv2.waitKey(0) & 0xFF == ord('n')
        with self.cv:
            while not k:
                self.cv.wait()
                k = cv2.waitKey(0) & 0xFF == ord('n') 

        for i in range(self.L):
            h = hgap + self.border//2
            for j in range(self.C):
                screen = np.ones((self.h, self.w), np.uint8)*255
                screen[v:v+size,h:h+size] = self.target
                h += hgap + size
                cv2.imshow(window, screen)
                if cv2.waitKey(2500) & 0xFF == ord('n'):
                    continue
            v += vgap + size
        cv2.destroyWindow(window)
        