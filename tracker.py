import numpy as np
import cv2

class Tracker():

    def __init__(self, confidence=0.85):
        self.centroids = np.empty((0,2), float)
        self.ratios    = np.empty((0,1), float)
        self.conf      = confidence


    def get_pupil(self, frame):
        '''
        Main method to track pupil position
        IN: BGR frame from live camera or video file
        OUT: ellipse 
        '''
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (9,9), 7)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
        ret, thresh = cv2.threshold(img, minVal+15, 255, cv2.THRESH_BINARY_INV)
        blob = self.__get_blob(thresh)
        cnt = self.__get_contours(blob) 
        cnt_e, ellipse = self.__get_ellipse(cnt, img)
        confidence = self.__get_confidence(cnt, cnt_e)
        if confidence > self.conf:
            return ellipse


    def __get_blob(self, bin_img):
        '''
        IN: thresholded image with pupil as foreground
        OUT: blob area containing only de pupil (hopefully)
        '''
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
        blob = np.zeros(bin_img.shape, np.uint8)
        stats = stats[1:]
        if len(stats) > 0:
            idx = np.argmax(stats[:,4]) + 1
            blob[labels==idx] = 255
            return blob


    def __get_contours(self, blob):
        '''
        IN: pupil blob in a binary image
        OUT: OpenCV contours of this blob 
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        closing = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
        cim, cnt, hiq = cv2.findContours(closing, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        return cv2.convexHull(cnt[0])

    
    def __get_ellipse(self, contour, img):
        '''
        IN: pupil contours and image frame
        OUT: fitted ellipse around pupil and its contours
        '''
        mask = np.zeros(img.shape, np.uint8)
        ellipse = None
        for c in [contour]:
            if len(c) >= 5:
                ellipse = cv2.fitEllipseDirect(c)
                break
        if ellipse is not None:
            mask = cv2.ellipse(mask, ellipse, 255, 2)
            cim, cnt, hiq = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)
            return cnt, ellipse

    
    def __get_confidence(blob_contour, ellipse_contour):
        '''
        Measures the rate of how certain we are that
        we found the pupil in a particular frame
        IN: original blob and actual fitted ellipse contours
        OUT: confidence index (0-1 float)
        '''
        blob_area = cv2.contourArea(blob_contour)
        ellipse_area = cv2.contourArea(ellipse_contour[0])
        return blob_area/ellipse_area

    