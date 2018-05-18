import cv2
import numpy as np
import marker_detector
import time
from threading import Thread


class SceneCamera(Thread):

    def __init__(self, video_src, width, height):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(video_src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.quit = False
        self.detector = marker_detector.MarkerDetector(width, height)
        self.code = [
            [1,1,1],
            [1,1,1],
            [1,1,0]
        ] 


    def run(self):
        while not self.quit:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                self.frame = None
            time.sleep(0.005)


    def get_marker_position(self):
        if self.frame is not None:
            return self.detector.detect(self.frame, self.code, True)
