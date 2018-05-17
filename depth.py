import cv2
import numpy as np 
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import kernels


class DepthEstimator():

    def __init__(self):
        self.calib_point = np.array((0,0), float)
        self.le_centers = np.empty((0,2), float)
        self.re_centers = np.empty((0,2), float)
        self.ids = np.empty((0,1), float)
        self.countdown = 7
        self.regressor = None

    
    def collect_data(self, target, leye, reye, plane_id):
        diff = np.abs(np.subtract(self.calib_point, target))
        if np.sum(diff) > 0.09:
            self.calib_point = target
            self.countdown = 7
        self.countdown -= 1
        if self.countdown <= 0:
            self.le_centers = np.vstack((self.le_centers, leye))
            self.re_centers = np.vstack((self.re_centers, reye))
            self.ids = np.vstack((self.ids, plane_id))


    def estimate_depth(self):
        kernel = 1.5*kernels.RBF(length_scale=1.0, length_scale_bounds=(0,3.0))
        clf = GaussianProcessClassifier(optimizer=None,
                                       n_restarts_optimizer=9,
                                       kernel = kernel)
        input_data = np.hstack((self.le_centers, self.re_centers))
        clf.fit(input_data, self.ids.ravel())
        self.regressor = clf


    def predict(self, leye, reye):
        if self.regressor is not None:
            input_data = np.hstack((leye, reye)).reshape(1,-1)
            plane_id = self.regressor.predict(input_data)[0]
            return plane_id




    
