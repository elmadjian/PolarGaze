import cv2
import sys
import numpy as np
import tracker
import polar
from matplotlib import pyplot as plt


if __name__=="__main__":
    if len(sys.argv) > 1:
        tracker = tracker.Tracker()
        pupil_action = polar.Polar()
        cap = cv2.VideoCapture(sys.argv[1])
        ring = None
        while True:
            ret, frame = cap.read()
            if ret:
                ellipse = tracker.find_pupil(frame)
                if ellipse is not None:
                    cv2.ellipse(frame, ellipse, (0,255,0), 2)
                if len(tracker.centroids) % 100 == 0:
                    ring = pupil_action.update_model(tracker.centroids)
                    print(ring)
                if ring is not None:
                    cv2.ellipse(frame, ring, (0,0,255), 2)
                cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

