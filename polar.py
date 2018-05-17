import cv2
import numpy as np 

class Polar():

    def __init__(self):
        self.center = None
        self.extremes = None
        self.angle = None
        self.major_axis = None
        self.minor_axis = None

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
        ellipse = cv2.fitEllipseDirect(cnt[0])
        if ellipse is not None:
            self.angle = ellipse[2]
            self.major_axis = ellipse[1][1]/2
            self.minor_axis = ellipse[1][0]/2
            self.center = ellipse[0]
            return ellipse

    
    def to_polar(self, vec):
        '''
        Converts cartesian to polar coordinates
        IN: numpy [] in cartesian coords
        OUT: numpy [] in polar coords
        '''
        rho = np.sqrt(vec[0]**2 + vec[1]**2)
        phi = np.arctan2(vec[1], vec[0])
        return np.array([rho, phi])


    def to_elliptical_space(self, vec, normalized=False):
        translated = np.subtract(vec, self.center)
        inverted = np.array([translated[0], -translated[1]])
        rotated = self.rotate(inverted)
        if normalized:
            nx = rotated[0]/self.minor_axis
            ny = rotated[1]/self.major_axis
            return np.array([nx, ny])
        return rotated

    
    # def distance(self, y_left_axis, vec):
    #     factor = y_left_axis + self.major_axis
    #     vec[1] = vec[1] + factor
    #     return vec


    def to_camera_space(self, vec):
        rotated = self.rotate(vec, -self.angle)
        inverted = np.array([rotated[0], -rotated[1]])
        translated = inverted + self.center
        return (int(translated[0]), int(translated[1]))

    
    def get_point_on_ellipse(self, vec):
        rotated = self.to_elliptical_space(vec)
        polar = self.to_polar(rotated)
        sx = np.sign(np.cos(polar[1]))
        sy = np.sign(np.sin(polar[1]))
        intersect = self.__ellipse_intersect(polar)
        point = np.array([intersect[0]*sx, intersect[1]*sy])
        return self.to_camera_space(point)


    def get_xy_edges(self, vec):
        rotated = self.to_elliptical_space(vec)
        px = self.__ellipse_intersect_x_axis(rotated)
        py = self.__ellipse_intersect_y_axis(rotated)
        trs_px = self.to_camera_space(px)
        trs_py = self.to_camera_space(py)
        return trs_px, trs_py


    def rotate(self, vec, deg=None):
        '''
        '''
        if deg is None:
            deg = self.angle
        theta = np.deg2rad(deg)
        R = np.array([
            [np.cos(theta),-np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return np.dot(R, vec)


    def __ellipse_intersect(self, polar):
        x = self.minor_axis * np.cos(polar[1])
        y = self.minor_axis * np.sin(polar[1])
        m = self.minor_axis
        M = self.major_axis
        tg2 = np.tan(polar[1])**2
        x_el = (m*M)/np.sqrt(M**2 + tg2 * m**2)
        y_el = (M/m)*np.sqrt(m**2 - x_el**2)
        return np.array([x_el, y_el])


    def __ellipse_intersect_x_axis(self, vec):
        a = self.major_axis
        b = self.minor_axis
        x = b/a * np.sqrt(np.abs(a**2 - vec[1]**2))
        x *= np.sign(vec[0])
        return np.array([x, vec[1]])


    def __ellipse_intersect_y_axis(self, vec):
        a = self.major_axis
        b = self.minor_axis
        y = a/b * np.sqrt(np.abs(b**2 - vec[0]**2))
        y *= np.sign(vec[1])
        return np.array([vec[0], y])


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
        

    