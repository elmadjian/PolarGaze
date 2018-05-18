import cv2
import numpy as np 
import pyrealsense as pyrs 


class RealSense():

    def __init__(self):
        self.position = np.array([-1,-1], np.uint8)
        self.coord3D = np.array([0,0,0], float)
        # serv = pyrs.Service()
        # color = pyrs.stream.ColorStream(fps=60, color_format='rgb')
        # depth = pyrs.stream.DepthStream(fps=60)
        # self.cam  = serv.Device(streams=(depth,))


    def run(self):
        depth = pyrs.stream.DepthStream(fps=60)
        color = pyrs.stream.ColorStream(fps=60, color_format='bgr')
        with pyrs.Service() as serv:
            with serv.Device(streams=(depth, color)) as dev:
                dev.apply_ivcam_preset(0)
                while True:
                    if dev.poll_for_frame():
                        d = dev.depth
                        d = self.__normalize_depth(d)
                        c = dev.color
                        if self.position[0] != -1 and self.position[1] != -1:
                            p = dev.deproject_pixel_to_point(self.position, d)
                            print(p)
                        cv2.imshow('test', d)
                        cv2.imshow('test2', c)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break


    def set_target(self, target_pos):
        self.position = target_pos


    def __normalize_depth(self, frame):
        M = np.mean(frame)
        frame[frame > M] = 255
        new_frame = (frame/M) * 255.0
        return np.array(new_frame, np.uint8)



if __name__=="__main__":
    rs = RealSense()
    rs.run()

    


