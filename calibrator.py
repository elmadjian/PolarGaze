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

#---------------------------------------------------------------

class Calibrator_3D():

    def __init__(self):#TODO: add binocular as an option
        self.leyeball = np.array((-0.12, 0.08, -0.05), float)
        self.reyeball = np.array((-0.02, 0.08, -0.05), float)
        self.radius = 0.01225 #antropomorphic average
        self.calib_point = np.array((0,0,0), float)
        self.targets = np.empty((0,3), float)
        self.l_centers = np.empty((0,2), float)
        self.r_centers = np.empty((0,2), float)
        self.countdown = 9
        self.lmodel = None
        self.rmodel = None

    
    def collect_data(self, target, leye, reye):
        diff = np.abs(np.subtract(self.calib_point, target))
        if np.sum(diff) > 0.09:
            self.calib_point = target
            self.countdown = 9
        self.countdown -= 1
        if self.countdown <= 0:
            self.targets = np.vstack((self.targets, target))
            self.l_centers = np.vstack((self.l_centers, leye))
            self.r_centers = np.vstack((self.r_centers, reye))


    def estimate_gaze(self):
        lb, lc = self.leyeball, self.l_centers
        rb, rc = self.reyeball, self.r_centers
        self.lmodel = self.__build_model_one_eye(lb, lc)
        self.rmodel = self.__build_model_one_eye(rb, rc)


    def __build_model_one_eye(self, eyeball, centers):
        kernel = 1.5*kernels.RBF(length_scale=1.0, length_scale_bounds=(0,3.0))
        clf = GaussianProcessRegressor(alpha=1e-5,
                                       optimizer=None,
                                       n_restarts_optimizer=9,
                                       kernel = kernel)
        surface = np.empty((0,3), float)                                       
        for t in self.targets:
            t = t - eyeball
            t_norm = t/np.linalg.norm(t)
            s = t_norm * self.radius
            surface = np.vstack((surface, s))
        print('centers length:', len(centers), "surface length:", len(surface))
        clf.fit(centers, surface)
        return clf


    def get_gaze_vector(self, leye, reye):
        lcoord, rcoord = None, None
        if self.lmodel is not None:
            lcoord = self.__gaze_vector(leye, self.lmodel)
        if self.rmodel is not None:
            rcoord = self.__gaze_vector(reye, self.rmodel)
        return lcoord, rcoord


    def __gaze_vector(self, eye, model):
        input_data = eye.reshape(1,-1)
        coord = model.predict(input_data)[0]
        return coord

        

