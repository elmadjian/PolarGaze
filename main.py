import cv2
import sys
import numpy as np
import tracker
import polar
from matplotlib import pyplot as plt


PREV_AREA = []

# main program
#=============
def main(video_source = "pupil.mkv"):
    cap = cv2.VideoCapture(video_source)
    centroids = np.empty((0,2), float)
    ratios    = np.empty((0,1), float)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            confidence, ellipse = find_pupil(gray)
            if confidence > 0.85:
                cv2.ellipse(frame, ellipse, (0,255,0), 2)
                ratio = ellipse[1][1]/ellipse[1][0]
                centroids = np.vstack((centroids, ellipse[0]))
                ratios = np.vstack((ratios, ratio))
            cv2.imshow("test", frame)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            mean = np.mean(centroids, axis=0)
            centers = centroids - mean
            polar = to_polar(centers)
            polar2 = polar * ratios
            polar2[:,1] = polar[:,1]


            cart = to_cartesian(polar)
            cart2 = to_cartesian(polar2)

            extremes = find_extremes(polar)
            cartesian = to_cartesian(extremes) + mean
            cnt = [np.array(cartesian, np.int32)]
            ring = cv2.fitEllipseDirect(cnt[0])
            if ring is not None:
                cv2.ellipse(frame, ring, (0,0,255), 5)
                cv2.imshow('testinho', frame)

            plt.scatter(cart[:,0], cart[:,1])
            plt.scatter(cart2[:,0], cart2[:,1], c='g')
            plt.scatter(0, 0, c='r')
            plt.show()
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     # plt.scatter(centroids[:,0], centroids[:,1])
        #     # mean = np.mean(centroids, axis=0)
        #     # plt.scatter(mean[0], mean[1], c='r')
        #     # plt.show()
        #     break

    cap.release()
    cv2.destroyAllWindows()

# find extreme points
#====================
def find_extremes(polar):
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

# transform to polar coordinates
#===============================
def to_polar(vec):
    rho = np.sqrt(vec[:,0]**2 + vec[:,1]**2).reshape((-1,1))
    phi = np.arctan2(vec[:,1], vec[:,0]).reshape((-1,1))
    coord = np.hstack((rho, phi))
    return coord


# transform to cartesian coords
#==============================
def to_cartesian(vec):
    x = (vec[:,0] * np.cos(vec[:,1])).reshape((-1,1))
    y = (vec[:,0] * np.sin(vec[:,1])).reshape((-1,1))
    coord = np.hstack((x,y))
    return coord


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
    return 0, None


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
        tracker = tracker.Tracker()
        pupil_action = polar.Polar()
        cap = cv2.VideoCapture(sys.argv[1])
        ring = None
        while True:
            ret, frame = cap.read()
            if ret:
                ellipse = tracker.find_pupil(frame)
                if ellipse is not None:
                    cv2.ellipse(frame, ellipse, (0,255,0), 2)
                if len(tracker.centroids) % 100 == 0:
                    ring = pupil_action.update_model(tracker.centroids)
                if ring is not None:
                    cv2.ellipse(frame, ring, (0,0,255), 2)
                cv2.imshow('test', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    else:
        main()

