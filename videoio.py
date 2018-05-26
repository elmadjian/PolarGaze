import subprocess
import re

class VideoIO():

    def __init__(self):
        self.eye_id_1 = None
        self.eye_id_2 = None
        self.rs_id = None
        self.read_inputs()

    
    def read_inputs(self):
        '''
        Only works for v4l2.
        No working solution for Mac or Windows yet
        '''
        binout = subprocess.check_output(["v4l2-ctl", "--list-devices"])
        out = binout.decode().split('\n')
        for i in range(len(out)):
            if re.findall('Integrated Camera', out[i]):
                eye_id = re.findall("video(\\d+)", out[i+1])
                if self.eye_id_1 is None:
                    self.eye_id_1 = int(eye_id[0])
                else:
                    self.eye_id_2 = int(eye_id[0])
            if re.findall('RealSense', out[i]):
                rs_id = re.findall("video(\\d+)", out[i+3])
                self._rs_id = int(rs_id[0])


    def get_rs_id(self):
        if self.rs_id is not None:
            return self.rs_id
        print("Could not find RealSense camera")

    
    def get_eye_id(self, side):
        if side == 'left':
            if self.eye_id_1 is not None:
                return self.eye_id_1
        elif side == 'right':
            if self.eye_id_2 is not None:
                return self.eye_id_2
        print(side.capitalize() + " eye camera is not plugged in")
            


if __name__=="__main__":
    v = VideoIO()
    v.read_inputs()