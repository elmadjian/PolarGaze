import cv2
import sys
import numpy as np
import tracker
import polar
import control
import eye
import time
from matplotlib import pyplot as plt


def process_eye_frame(tracker, polar, frame):
    ellipse = tracker.find_pupil(frame)
    if ellipse is not None:
        cv2.ellipse(frame, ellipse, (0,255,0), 2)
    if len(tracker.centroids) % 50 == 0:
        ring = polar.update_model(tracker.centroids)
        tracker.update_centroids(polar.extremes)
        return ring
    
def show_action(ring, frame, side):
    if ring is not None:
        cv2.ellipse(frame, ring, (0,0,255), 2)
    cv2.imshow(side, frame)


if __name__=="__main__":
    control    = control.Control()
    le_tracker = tracker.Tracker()
    re_tracker = tracker.Tracker()
    le_pupil   = polar.Polar()
    re_pupil   = polar.Polar()
    left_eye   = eye.Eye(le_tracker, le_pupil, 'left_eye.avi')
    right_eye  = eye.Eye(re_tracker, re_pupil, 'right_eye.avi')
    cap_scene  = cv2.VideoCapture('scene.avi')
    control.start()


    while True:
        sc_ret, sc_frame = cap_scene.read()

        if sc_ret:
            cv2.imshow('left', left_eye.get_frame(control.action))
            cv2.imshow('right', right_eye.get_frame(control.action))
            cv2.imshow('scene', sc_frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    control.join()
    cv2.destroyAllWindows()

