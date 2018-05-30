import cv2
import numpy as np
from threading import Thread

left = 4
right = 5
scene = 3

def record():
    cam1 = cv2.VideoCapture(left)
    cam2 = cv2.VideoCapture(right)
    cam3 = cv2.VideoCapture(scene)

    set_properties(cam1, 30.0, 800, 600)
    set_properties(cam2, 30.0, 800, 600)
    set_properties(cam3, 30.0, 1280, 720)

    codec = cv2.VideoWriter_fourcc(*'XVID')
    le_out = cv2.VideoWriter('left_eye.avi', codec, 30.0, (800,600))
    re_out = cv2.VideoWriter('right_eye.avi', codec, 30.0, (800,600))
    sc_out = cv2.VideoWriter('scene.avi', codec, 30.0, (1280, 720))

    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        ret3, frame3 = cam3.read()
        if ret1:
            cv2.imshow('left', frame1)
            le_out.write(frame1)
        if ret2:
            cv2.imshow('right', frame2)
            re_out.write(frame2)
        if ret3:
            cv2.imshow('scene', frame3)
            sc_out.write(frame3)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

def set_properties(cam, fps, width, height):
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    


if __name__=='__main__':
    record()