import evdev
from evdev import InputDevice, categorize, ecodes
from threading import Thread


class View(Thread):

    def __init__(self, controller):
        Thread.__init__(self)
        self.calibration = False
        self.active = False
        self.controller = controller


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

        calibrations = [i for i in range(2,12)]

        if device is not None:
            for event in device.read_loop():
                if event.type == ecodes.EV_KEY:
                    #print(event.code)
                    if event.code == 46 and event.value == 1: #'c'
                        self.calibration = not self.calibration
                        if self.calibration:
                            print('Please, choose a number [1-9]:')
                        else:
                            print("finished calibration")
                            self.controller.end_calibration()
                   
                    if event.code == 31 and event.value == 1: #'s'
                        print("detecting pupil action area")
                        self.controller.build_model()

                    if event.code == 19 and event.value == 1: #'r'
                        print('resetting model')
                        self.controller.reset_model()

                    if event.code in calibrations and event.value == 1: #'0-9'
                        '''
                        0: inactive
                        1-9: calibration id
                        '''
                        id = event.code -1
                        if 42 in device.active_keys() or 54 in device.active_keys(): #SHIFT + ...
                            if id < 10:
                                print('using calibration', event.code-1)
                            self.controller.use_calibration(event.code-1)
                        if self.calibration and 56 not in device.active_keys():
                            print("Calibrating for id:", id)
                            self.controller.calibrate(id)

                    if event.code == 16 and event.value == 1: #'q'
                        print('quitting...')
                        break
        else:
            print("ALERT: No keyboard detected.")
