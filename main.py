import cv2
import sys
import numpy as np
import tracker
import polar
import view
import eye
import marker_detector
import calibrator
import depth
import time
import calib_screen
import realsense
import threading
import scene
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
        self.d_estimator  = depth.DepthEstimator()
        self.calibrations = {i:[None,None] for i in range(1,10)}
        self.calibrating  = False
        self.active   = False
        self.__setup_video_input(argv)
        self.left_e   = eye.Eye(self.le_track, self.polar_le, self.le_video)
        self.right_e  = eye.Eye(self.re_track, self.polar_re, self.re_video)
        self.marker = cv2.imread('marker2.png', cv2.IMREAD_GRAYSCALE)
        self.screen = None
        self.cv = threading.Condition()


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
        calib_left  = calibrator.Calibrator(1280,720,binocular=False)
        calib_right = calibrator.Calibrator(1280,720,binocular=False) 
        self.calibrations[id][0] = calib_left
        self.calibrations[id][1] = calib_right
        self.screen = calib_screen.CalibrationScreen(1920,1080,3,4,
                                            self.marker, self.cv)
        self.screen.start()


    def end_calibration(self):
        id = self.calibrating
        self.calibrations[id][0].estimate_gaze()
        self.calibrations[id][1].estimate_gaze()
        #self.d_estimator.estimate_depth()
        self.calibrating = False
        self.screen.join()

    
    def use_calibration(self, id):
        if id in self.calibrations.keys():
            if self.calibrations[id][0] is not None:
                self.active = id
            else:
                print('No calibration has been found for id:', id)
        else:
            self.active = False


    def __collect_data(self, target, leye, reye):
        id = self.calibrating
        if target is not None and leye is not None and reye is not None:
            self.calibrations[id][0].collect_data(target, leye)
            self.calibrations[id][1].collect_data(target, reye)
            #self.d_estimator.collect_data(target, leye, reye, id)



    def run(self):
        scn = scene.SceneCamera(self.sc_video) 
        kbd = view.View(self, self.cv)
        scn.start()
        kbd.start()

        while True:
            if scn.frame is not None:
                cv2.imshow('left', self.left_e.get_frame('l'))
                cv2.imshow('right', self.right_e.get_frame('r'))
                le_c = self.left_e.centroid
                re_c = self.right_e.centroid
                if self.calibrating:
                    target = scn.get_marker_position()
                    self.__collect_data(target, le_c, re_c)
                elif self.active:
                    lcoord = self.calibrations[self.active][0].predict(le_c)
                    rcoord = self.calibrations[self.active][1].predict(re_c)
                    #plane_id = self.d_estimator.predict(le_c, re_c)
                    #print(plane_id)
                    if lcoord is not None and rcoord is not None:
                        cv2.circle(scn.frame, lcoord, 12, (200,0,200),-1)
                        cv2.circle(scn.frame, rcoord, 12, (0,200,200),-1)
                cv2.imshow('scene', scn.frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                scn.quit = True
                break

        kbd.join()
        scn.join()
        cv2.destroyAllWindows()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
#====================================================================
if __name__=="__main__":
    controller = Controller(sys.argv)
    controller.run()






