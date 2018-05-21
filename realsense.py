import cv2
import numpy as np
import pyrealsense as pyrs 
import marker_detector
import time
from threading import Thread


class RealSense(Thread):

    def __init__(self, width, height):
        Thread.__init__(self)
        self.position = None
        self.coord3D = None
        self.color_frame = None
        self.depth_frame = None
        self.quit = False
        self.detector = marker_detector.MarkerDetector(width, height)
        self.code = [
            [1,1,1],
            [1,1,1],
            [1,1,0]
        ]


    def run(self):
        dac   = pyrs.stream.DACStream(fps=60)
        depth = pyrs.stream.DepthStream(fps=60)
        color = pyrs.stream.ColorStream(fps=60, color_format='bgr')
        with pyrs.Service() as serv:
            with serv.Device(streams=(depth, color, dac)) as dev:
                dev.apply_ivcam_preset(0)
                while not self.quit:
                    if dev.poll_for_frame():
                        d = dev.dac
                        self.color_frame = dev.color
                        if self.position is not None:
                            self.coord3D = self.__find_3d_coord(self.position, d, dev)
                            self.position = None
                    time.sleep(0.005)


    def set_marker_position(self):
        if self.color_frame is not None:
            frame = self.color_frame
            self.position = self.detector.detect(frame, self.code)
            if self.position is None:
                self.coord3D = None


    def __normalize_depth(self, frame):
        M = np.mean(frame)
        new_frame = frame.copy()
        new_frame[new_frame > M] = 255
        new_frame = (new_frame/M) * 255.0
        return np.array(new_frame, np.uint8)


    def __find_3d_coord(self, position, depth, dev):
        x = position[1]
        y = position[0]
        est_depth = depth[x,y]
        pos = np.array([y,x], np.float32)
        if est_depth > 0.6:
            return dev.deproject_pixel_to_point(pos, est_depth)/1000.0
                       




if __name__=="__main__":
    rs = RealSense(640,480)
    rs.run()

    


