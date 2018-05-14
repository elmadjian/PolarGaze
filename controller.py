import evdev
from evdev import InputDevice, categorize, ecodes
from threading import Thread


class Control(Thread):

    def __init__(self, calibrator):
        Thread.__init__(self)
        self.action = False
        self.calibration = False
        self.estimation = False
        self.calibrated = False


    def run(self):
        '''
        Callback for keyboard
        's' KEY is used to toggle pupil action area mapping
        'c' KEY is used to trigger calibration procedure
        'q' KEY is used to quit
        '''
        device = None
        devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
        for d in devices:
            if "keyboard" in d.name:
                device = d
                break

        if device is not None:
            for event in device.read_loop():
                if event.type == ecodes.EV_KEY:
                    if event.code == 46 and event.value == 1:
                        self.calibration = not self.calibration
                        if self.calibration:
                            print("calibrating...")
                            self.calibrated = False
                            self.estimation = False
                        else:
                            print("finishing calibration")
                            self.estimation = True
                            self.calibrated = True
                    if event.code == 31 and event.value == 1:
                        self.action = not self.action
                        if self.action:
                            print("detecting pupil action area")
                        else:
                            print("ending detection")
                    if event.code == 16 and event.value == 1:
                        print('quitting...')
                        break
        else:
            print("ALERT: No keyboard detected.")
