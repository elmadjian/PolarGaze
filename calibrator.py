import cv2
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

class Calibrator():

    def __init__(self, binocular=True, in3d=False):
        self.calib_point = np.array((0,0), float)
        self.targets = np.empty((0,2), float)
        if in3d:
            self.targets = np.empty((0,3), float)
            self.calib_point = np.array((0,0,0), float)
        self.l_centers = np.empty((0,2), float)
        self.r_centers = np.empty((0,2), float)
        self.countdown = 9
        self.regressor = None
        self.binocular = binocular


    def collect_data(self, target, leye, reye=None):
        diff = np.abs(np.subtract(self.calib_point, target))
        if np.sum(diff) > 0.09:
            self.calib_point = target
            self.countdown = 9
        self.countdown -= 1
        if self.countdown <= 0:
            self.targets = np.vstack((self.targets, target))
            self.l_centers = np.vstack((self.l_centers, leye))
            if reye is not None:
                self.r_centers = np.vstack((self.r_centers, reye))


    def estimate_gaze(self):
        kernel = 1.5*kernels.RBF(length_scale=1.0, length_scale_bounds=(0,3.0))
        clf = GaussianProcessRegressor(alpha=1e-5,
                                       optimizer=None,
                                       n_restarts_optimizer=9,
                                       kernel = kernel)
        if self.binocular:
            input_data = np.hstack((self.l_centers, self.r_centers))
            clf.fit(input_data, self.targets)
        else:
            clf.fit(self.l_centers, self.targets)
        self.regressor = clf


    def predict(self, leye, reye=None, w=None, h=None):
        if self.regressor is not None:
            input_data = leye.reshape(1,-1)
            if reye is not None:
                input_data = np.hstack((leye, reye))
            coord = self.regressor.predict(input_data)[0]
            x = coord[0] * w
            y = coord[1] * h
            if len(coord) == 3:
                return coord 
            return (int(x), int(y))

        

