import cv2
import numpy as np 


class Eye():

    def __init__(self, tracker, polar, feed):
        self.tracker = tracker
        self.polar = polar
        self.cap = cv2.VideoCapture(feed)
        self.ring = None


    def get_frame(self, detect_action):
        ret, frame = self.cap.read()
        if ret:
            self.process_frame(frame)
            if detect_action:
                self.process_action(frame)
            return self.post_frame(frame)
        return False, None
            
    
    def process_frame(self, frame):
        ellipse = self.tracker.find_pupil(frame)
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0,255,0), 2)


    def process_action(self, frame):
        if len(self.tracker.centroids) % 50 == 0:
            self.ring = self.polar.update_model(self.tracker.centroids)
            self.tracker.update_centroids(self.polar.extremes)

    
    def post_frame(self, frame):
        if self.ring is not None:
            cv2.ellipse(frame, self.ring, (0,0,255), 2)
        return frame