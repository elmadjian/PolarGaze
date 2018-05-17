import cv2
import sys
import numpy as np
import tracker
import polar
import view
import eye
import marker_detector
import calibrator
import time
from matplotlib import pyplot as plt


class Controller():

    def __init__(self, argv):
        self.le_video = None
        self.re_video = None
        self.sc_video = None
        self.polar_le = polar.Polar()
        self.polar_re = polar.Polar()
        self.le_track = tracker.Tracker()
        self.re_track = tracker.Tracker()
        self.calibrations = {i:None for i in range(1,10)}
        self.calibrating  = False
        self.active   = False
        self.__setup_video_input(argv)
        self.left_e   = eye.Eye(self.le_track, self.polar_le, self.le_video)
        self.right_e  = eye.Eye(self.re_track, self.polar_re, self.re_video)


    def __setup_video_input(self, argv):
        if len(argv) > 1:
            self.le_video = int(argv[1])
            self.re_video = int(argv[2])
            self.sc_video = int(argv[3])
        else:
            self.le_video = 'videos/left003.avi'
            self.re_video = 'videos/right003.avi'
            self.sc_video = 'videos/scene003.avi'


    def build_model(self):
        le_ring = self.polar_le.update_model(self.le_track.centroids)
        re_ring = self.polar_re.update_model(self.re_track.centroids)
        self.le_track.update_centroids(self.polar_le.extremes)
        self.re_track.update_centroids(self.polar_re.extremes)
        self.left_e.ring = le_ring
        self.right_e.ring = re_ring


    def reset_model(self):
        self.polar_le.extremes = None
        self.polar_re.extremes = None
        self.le_track.centroids = np.empty((0,2), float)
        self.re_track.centroids = np.empty((0,2), float)


    def calibrate(self, id):
        self.calibrating = id
        calibration = calibrator.Calibrator()
        self.calibrations[id] = calibration


    def end_calibration(self):
        id = self.calibrating
        self.calibrations[id].estimate_gaze()
        self.calibrating = False

    
    def use_calibration(self, id):
        if id in self.calibrations.keys():
            if self.calibrations[id] is not None:
                self.active = id
            else:
                print('No calibration has been found for id:', id)
        else:
            self.active = False


    def __collect_data(self, frame, code, detector, leye, reye):
        id = self.calibrating
        target = detector.detect(frame, code)
        self.calibrations[id].collect_data(target, leye, reye)



    def run(self):
        kbd       = view.View(self)
        detector  = marker_detector.MarkerDetector()
        cap_scene = cv2.VideoCapture(self.sc_video)
        cap_scene.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap_scene.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        code = [
            [1,1,1],
            [1,0,0],
            [1,0,0]
        ]
        kbd.start()

        while True:
            sc_ret, sc_frame = cap_scene.read()
            if sc_ret:
                cv2.imshow('left', self.left_e.get_frame('l'))
                cv2.imshow('right', self.right_e.get_frame('r'))
                le_c = self.left_e.centroid
                re_c = self.right_e.centroid
                if self.calibrating:
                    self.__collect_data(sc_frame, code, detector, le_c, re_c)
                elif self.active:
                    coord = self.calibrations[self.active].predict(le_c, re_c)
                    if coord is not None:
                        pos = (int(coord[0]), int(coord[1]))
                        cv2.circle(sc_frame, pos, 12, (200,0,200),-1)
                cv2.imshow('scene', sc_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        kbd.join()
        cv2.destroyAllWindows()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#====================================================================
if __name__=="__main__":
    controller = Controller(sys.argv)
    controller.run()






