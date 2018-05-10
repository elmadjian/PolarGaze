import cv2
import sys
import numpy as np
import queue
from matplotlib import pyplot as plt


PREV_AREA = []

# main program
#=============
def main(vide_source = "pupil.mkv"):
    cap = cv2.VideoCapture(vide_source)
    pupil_hist = []

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            find_pupil(gray)
            # painted = remove_glare(gray)
            # hist = cv2.calcHist([painted], [0], None, [256], [0,256])
            # cut  = find_pupil(hist, pupil_hist)
            # print(cut)
            # if len(pupil_hist) > 5:
            #     pupil_hist.pop(0)
            # blob = get_blob(painted, cut)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # ellipse = find_ellipse(blob)
            # if ellipse is not None:
            #     cv2.ellipse(color, ellipse, (0,255,0), 2)
            cv2.imshow("test", color)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# remove bright spots
#====================
def remove_glare(img):
    blur = cv2.GaussianBlur(img, (9,9), 7)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
    mask = np.zeros(img.shape, dtype="uint8")
    cv2.circle(mask, maxLoc, 15, (255,255,255), -1)
    return cv2.inpaint(img, mask, 15, cv2.INPAINT_TELEA)


# find pupil and confidence
#=====================
def find_pupil(img):
    blur = cv2.GaussianBlur(img, (9,9), 7)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(blur)
    ret, thresh = cv2.threshold(img, minVal+15, 255, cv2.THRESH_BINARY_INV)
    ret, blob = get_blob(thresh)
    if ret:
        cnt = get_contours(blob) 
        cnt_e, ellipse = get_ellipse(cnt, img)
        confidence = get_confidence(cnt, cnt_e)
        return confidence, ellipse
    return None, None


#get the biggest continuous blob
#===============================
def get_blob(img):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
    stats = stats[1:]
    max_area, idx, mean = 0, 0, 0
    if PREV_AREA:
        mean = np.mean(PREV_AREA)
    if len(stats) > 0:
        for i in range(len(stats)):
            p1 = np.array([stats[i][0],stats[i][1]])
            p2 = np.array([stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]])
            d = np.linalg.norm(p1-p2)
            if not PREV_AREA:
                mean = d
            if d > max_area:
                if 0.7 < d/mean < 1.3:
                    max_area = d
                    idx = i+1
        if max_area != 0:
            PREV_AREA.append(max_area)
            img[labels==idx] = 255
            img[labels!=idx] = 0
            if len(PREV_AREA) > 5:
                PREV_AREA.pop(0)
            return True, img
    return False, np.zeros(img.shape, np.uint8)


# get blob contours
#==================
def get_contours(blob):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    closing = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
    cim, cnt, hiq = cv2.findContours(closing, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    return cv2.convexHull(cnt[0])


#get the ellipse and its contours
#================================
def get_ellipse(contour, img):
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

# get estimate of fitting
#========================
def get_confidence(contour, ellipse):
    hull_area = cv2.contourArea(contour)
    ellipse_area = cv2.contourArea(ellipse[0])
    solidity = hull_area/ellipse_area
    return solidity



if __name__=="__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()




# def __grow_area(seed, img):
#     q = queue.Queue()
#     mask = np.zeros(img.shape, np.uint8)
#     pinv = (seed[1], seed[0])
#     mask[pinv] = 255
#     q.put(pinv)
#     while not q.empty():
#         p = q.get()
#         p1 = (p[0]-1, p[1])
#         p2 = (p[0]+1, p[1])
#         p3 = (p[0], p[1]+1)
#         p4 = (p[0], p[1]-1)
#         plist = [p1,p2,p3,p4]
#         for el in plist:
#             if (abs(img[p] - img[el]) < 40):
#                 if img[el] != 255 and mask[el] != 255:
#                     mask[el] = 255
#                     q.put(el)
#         cv2.imshow('testinho', mask)
#         cv2.imshow('testinho2', img)
#         cv2.waitKey(0)