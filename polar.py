import cv2
import numpy as np 

class Polar():

    def __init__(self):
        self.mean = 0

    def update_model(self, centroids):
        mean = np.mean(centroids, axis=0)
        ellipse_centers = centroids - mean
        polar = self.to_polar_batch(ellipse_centers)
        extremes = self.find_extremes(polar)
        cartesian = self.to_cartesian_batch(extremes) + mean
        cnt = [np.array(cartesian, np.int32)]
        action_area = cv2.fitEllipseDirect(cnt[0])
        return action_area

    
    def to_polar_batch(self, vec):
        '''
        Converts cartesian to polar coordinates
        IN: [[]..[]] in cartesian coords
        OUT: [[]..[]] in polar coords (rho, phi)
        '''
        rho = np.sqrt(vec[:,0]**2 + vec[:,1]**2).reshape((-1,1))
        phi = np.arctan2(vec[:,1], vec[:,0]).reshape((-1,1))
        coord = np.hstack((rho, phi))
        return coord


    def to_cartesian_batch(self, vec):
        '''
        Converts polar to cartesian coordinates
        IN: [[]..[]] in polar (rho, phi) coords
        OUT: [[]..[]] in cartesian (x,y) coords
        '''
        x = (vec[:,0] * np.cos(vec[:,1])).reshape((-1,1))
        y = (vec[:,0] * np.sin(vec[:,1])).reshape((-1,1))
        coord = np.hstack((x,y))
        return coord


    def find_extremes(self, polar):
        '''
        Finds the outermost points of pupil displacement
        IN: polar coords of all pupil displacements
        OUT: list of 12 extreme points
        '''
        ordered  = polar[polar[:,1].argsort()]
        step     = len(ordered)//12
        extremes = np.empty((0,2))
        for i in range(0, len(ordered), step):
            searchable = np.empty((0,2))
            for j in range(i, i+step):
                if j == len(ordered):
                    break
                searchable = np.vstack((searchable, ordered[j]))
            extremes = np.vstack((extremes, np.max(searchable, axis=0)))
        return extremes
        

    