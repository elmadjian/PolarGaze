import cv2
import numpy as np
from threading import Thread

in1 = 1
in2 = 6
in3 = 5

def record():
    cam1 = cv2.VideoCapture(in1)
    cam2 = cv2.VideoCapture(in2)
    cam3 = cv2.VideoCapture(in3)

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        ret3, frame3 = cam3.read()
        if ret1:
            cv2.imshow('left', frame1)
        if ret2:
            cv2.imshow('right', frame2)
        if ret3:
            cv2.imshow('scene', frame3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    


if __name__=='__main__':
    record()