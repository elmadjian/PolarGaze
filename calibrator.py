import cv2
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

class Calibrator():

    def __init__(self):
        self.calib_point = np.array((0,0), float)
        self.targets = np.empty((0,2), float)
        self.le_centers = np.empty((0,2), float)
        self.re_centers = np.empty((0,2), float)
        self.countdown = 7
        self.regressor = None


    def collect_data(self, target, leye, reye):
        if self.calib_point is not None and target is not None:
            diff = np.abs(np.subtract(self.calib_point, target))
            if np.sum(diff) > 0.09:
                self.calib_point = target
                self.countdown = 7
            self.countdown -= 1
            if self.countdown <= 0:
                self.targets = np.vstack((self.targets, target))
                self.le_centers = np.vstack((self.le_centers, leye))
                self.re_centers = np.vstack((self.re_centers, reye))


    def estimate_gaze(self):
        kernel = 1.5*kernels.RBF(length_scale=1.0, length_scale_bounds=(0,3.0))
        clf = GaussianProcessRegressor(alpha=1e-5,
                                       optimizer=None,
                                       n_restarts_optimizer=9,
                                       kernel = kernel)
        input_data = np.hstack((self.le_centers, self.re_centers))
        clf.fit(input_data, self.targets)
        self.regressor = clf


    def predict(self, leye, reye):
        if self.regressor is not None:
            input_data = np.hstack((leye, reye)).reshape(1,-1)
            coord = self.regressor.predict(input_data)[0]
            x = coord[0] * 1280
            y = coord[1] * 720
            return np.array([x,y])

        

