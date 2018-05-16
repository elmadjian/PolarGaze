import cv2
import numpy as np 


class Eye():

    def __init__(self, tracker, polar, feed):
        self.tracker = tracker
        self.polar = polar
        self.cap = cv2.VideoCapture(feed)
        self.ring = None
        self.centroid = None


    def get_frame(self, side=None):
        ret, frame = self.cap.read()
        if ret:
            if side == 'l':
                frame = cv2.flip(frame, 0)
            elif side == 'r':
                frame = cv2.flip(frame, 1)
            self.__process_frame(frame)
            return self.__post_frame(frame)
        return False, None
            
    
    def __process_frame(self, frame):
        ellipse = self.tracker.find_pupil(frame)
        if ellipse is not None:
            cv2.ellipse(frame, ellipse, (0,255,0), 2)
            if self.ring is not None:
                excentricity = ellipse[1][1]/ellipse[1][0]
                translated = np.subtract(ellipse[0], self.polar.center)
                inverted = np.array([translated[0], -translated[1]])
                rotated = self.polar.rotate(inverted)
                # x = rotated[0]/self.polar.minor_axis# * excentricity
                # y = rotated[1]/self.polar.major_axis# * excentricity
                x = ellipse[0][0]/800
                y = ellipse[0][1]/600
                self.centroid = np.array([x,y], float)
                self.__draw_ellipse_axes(frame, self.ring)
                #print(self.centroid)


    def __post_frame(self, frame):
        if self.ring is not None:
            cv2.ellipse(frame, self.ring, (0,0,255), 2)
        return frame


    def __draw_ellipse_axes(self, frame, ring):
        cos = np.cos(np.deg2rad(ring[2]))
        sin = np.sin(np.deg2rad(ring[2]))
        la_x0 = int(ring[0][0] - ring[1][0]/2 * cos)
        la_y0 = int(ring[0][1] - ring[1][0]/2 * sin)
        la_x1 = int(ring[0][0] + ring[1][0]/2 * cos)
        la_y1 = int(ring[0][1] + ring[1][0]/2 * sin)
        sa_x0 = int(ring[0][0] - ring[1][1]/2 * sin)
        sa_y0 = int(ring[0][1] + ring[1][1]/2 * cos)
        sa_x1 = int(ring[0][0] + ring[1][1]/2 * sin)
        sa_y1 = int(ring[0][1] - ring[1][1]/2 * cos)
        cv2.line(frame, (la_x0, la_y0), (la_x1, la_y1), (0,0,255), 1)
        cv2.line(frame, (sa_x0, sa_y0), (sa_x1, sa_y1), (0,0,255), 1)