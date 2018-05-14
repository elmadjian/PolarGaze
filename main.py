import cv2
import sys
import numpy as np
import tracker
import polar
import controller
import eye
import marker_detector
import calibrator
from matplotlib import pyplot as plt


if __name__=="__main__":
    control    = controller.Control()
    calibrator = calibrator.Calibrator()
    le_tracker = tracker.Tracker()
    re_tracker = tracker.Tracker()
    le_pupil   = polar.Polar()
    re_pupil   = polar.Polar()
    detector   = marker_detector.MarkerDetector()
    left_eye   = eye.Eye(le_tracker, le_pupil, 'left_eye.avi')
    right_eye  = eye.Eye(re_tracker, re_pupil, 'right_eye.avi')
    cap_scene  = cv2.VideoCapture('scene.avi')
    code = [
        [1,1,1],
        [1,0,0],
        [1,0,0]
    ]
    control.start()


    while True:
        sc_ret, sc_frame = cap_scene.read()

        if sc_ret:
            cv2.imshow('left', left_eye.get_frame(control.action))
            cv2.imshow('right', right_eye.get_frame(control.action))
            center = detector.detect(sc_frame, code)
            cv2.imshow('scene', sc_frame)
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    control.join()
    cv2.destroyAllWindows()

