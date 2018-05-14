import cv2
import numpy as np 

class Polar():

    def __init__(self):
        self.center = None
        self.extremes = None

    def update_model(self, centroids):
        '''
        Update the area of pupil displacement
        IN: list of projected pupil centroids
        OUT: ellipse of the area
        '''
        mean = np.mean(centroids, axis=0)
        ellipse_centers = centroids - mean
        polar = self.__to_polar_batch(ellipse_centers)
        extremes = self.__find_extremes(polar, 18)
        cartesian = self.__to_cartesian_batch(extremes) + mean
        self.extremes = cartesian
        cnt = [np.array(cartesian, np.int32)]
        action_area = cv2.fitEllipseDirect(cnt[0])
        if action_area is not None:
            self.center = action_area[0]
            return action_area

    
    def to_polar(self, vec):
        '''
        Converts cartesian to polar coordinates
        IN: numpy [] in cartesian coords
        OUT: numpy [] in polar coords
        '''
        rho = np.sqrt(vec[0]**2 + vec[1]**2)
        phi = np.arctan2(vec[1], vec[0])
        return np.array([rho, phi])


    def __to_polar_batch(self, vec):
        '''
        Converts cartesian to polar coordinates in batches
        IN: [[]..[]] in cartesian coords
        OUT: [[]..[]] in polar coords (rho, phi)
        '''
        rho = np.sqrt(vec[:,0]**2 + vec[:,1]**2).reshape((-1,1))
        phi = np.arctan2(vec[:,1], vec[:,0]).reshape((-1,1))
        return np.hstack((rho, phi))


    def __to_cartesian_batch(self, vec):
        '''
        Converts polar to cartesian coordinates in batches
        IN: [[]..[]] in polar (rho, phi) coords
        OUT: [[]..[]] in cartesian (x,y) coords
        '''
        x = (vec[:,0] * np.cos(vec[:,1])).reshape((-1,1))
        y = (vec[:,0] * np.sin(vec[:,1])).reshape((-1,1))
        return np.hstack((x,y))


    def __find_extremes(self, polar, s):
        '''
        Finds the outermost points of pupil displacement
        IN: polar coords of all pupil displacements and how many samples
        OUT: list of s extreme points
        '''
        ordered  = polar[polar[:,1].argsort()]
        step     = len(ordered)//s
        extremes = np.empty((0,2))
        for i in range(s):
            searchable = ordered[i*step:i*step+step,:]
            if i == s-1:
                searchable = ordered[i*step:,:]
            idx = np.argmax(searchable, axis=0)[0]
            extremes = np.vstack((extremes, searchable[idx]))
        return extremes
        

    