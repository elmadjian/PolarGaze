import cv2, sys
import numpy as np

class MarkerDetector():

    def __init__(self, width, height, calibration=None):
        self.calibration = calibration
        self.gray = None
        self.width = width
        self.height = height


    def detect(self, img, code, normalized=False):
        code = np.array(code).reshape((3,3)).astype("bool")
        preprocessed_img = self._preprocess(img)
        contour_list     = self._find_contours(preprocessed_img)
        possible_markers = self._find_candidates(contour_list, img)
        transformed_ones = self._transform_marker(possible_markers, img)
        ret, center      = self._get_marker_code(transformed_ones, code, img)
        if ret:
            if normalized:
                x = center[0]/1280
                y = center[1]/720
                return np.array([x,y], float)
            return np.array([center[0], center[1]], int)
        # if ret:
        #     return self._get_vectors(corners)
        # return None, None


    def _preprocess(self, img):
        filtered_img = cv2.medianBlur(img, 5)
        self.gray    = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        # return cv2.adaptiveThreshold(self.gray, 255,
        #                               cv2.ADAPTIVE_THRESH_MEAN_C,
        #                               cv2.THRESH_BINARY, 9, 9)
        return cv2.threshold(self.gray, 0, 255, cv2.THRESH_OTSU)[1]


    def _find_contours(self, img):
        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
        contour_list = []
        for c in contours:
            if len(c) > 5:
                contour_list.append(c)
        return contour_list


    def _find_candidates(self, contour_list, img):
        possible_markers = []
        for c in contour_list:
            eps = cv2.arcLength(c, True) * 0.05
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue
            if self._is_min_length(approx, 1000):
                continue
            v1 = (approx[1] - approx[0])[0]
            v2 = (approx[2] - approx[0])[0]
            lr = (v1[0] * v2[1]) - (v1[1] * v2[0])
            if lr < 0.0:
                approx[1], approx[3] = approx[3], approx[1]
            possible_markers.append(approx)
        return possible_markers


    def _is_min_length(self, approx, min_length):
        min_dist = sys.maxsize
        for i in range(4):
            points = approx[i] - approx[(i+1)%4]
            squared_length = np.dot(points, points.T)
            min_dist = min(min_dist, squared_length[0][0])
        if min_dist < min_length:
            return True
        return False


    def _transform_marker(self, candidates, img):
        transformed = []
        for m in candidates:
            (m0, m1, m2, m3) = m
            m0, m1, m2, m3 = m0[0], m1[0], m2[0], m3[0]
            new_m = np.array([m0, m1, m2, m3], dtype="float32")
            side0 = np.sqrt((m0[0]-m1[0])**2 + (m0[1]-m1[1])**2)
            side1 = np.sqrt((m1[0]-m2[0])**2 + (m1[1]-m2[1])**2)
            side2 = np.sqrt((m2[0]-m3[0])**2 + (m2[1]-m3[1])**2)
            side3 = np.sqrt((m3[0]-m0[0])**2 + (m3[1]-m0[1])**2)
            side  = int(max(side0, side1, side2, side3))
            dst = np.array([
                [0,0],
                [side-1, 0],
                [side-1, side-1],
                [0, side-1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(new_m, dst)
            warped = cv2.warpPerspective(img, M, (side, side))
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU)[1]
            transformed.append([thresh, m])
        return transformed


    def _get_marker_code(self, markers, code, img):
        for m in markers:
            transformed, contour = m[0], m[1]
            side = transformed.shape[0]
            step = side//5
            bit_matrix = np.zeros((5,5), dtype="bool")
            for y in range(5):
                for x in range(5):
                    ys, xs = y*step, x*step
                    cut = transformed[ys:ys+step, xs:xs+step]
                    cut[cut==255] = 1
                    if cut.sum() > (cut.shape[0]*cut.shape[1]) * 0.45:
                        bit_matrix[y,x] = True
            ret, rot = self._check_code(code, bit_matrix)
            if ret:
                contour = np.roll(contour, -(rot*2))
                cv2.polylines(img, [contour], True, (255,0,255), 2)
                cnt = [c[0] for c in contour]
                mean = np.mean(cnt, axis=0)
                center = (int(mean[0]), int(mean[1]))
                cv2.circle(img, center, 5, (100, 50, 255), -1)
                return True, center
        return False, []


    def _check_code(self, code, matrix):
        if matrix[0,:].any() or matrix[-1,:].any():
            return False, -1
        if matrix[:,0].any() or matrix[:,-1].any():
            return False, -1
        cut = matrix[1:-1, 1:-1]
        for i in range(4):
            if (cut == code).all():
                return True, i
            cut = np.rot90(cut)
        return False, -1


    def _get_vectors(self, corners):
        crit = self.calibration.criteria
        C    = self.calibration.C
        dist = self.calibration.dist
        imgp = np.array(corners, dtype="float32")
        objp = np.array([[-0.5, -0.5, 0.0],
                         [ 0.5, -0.5, 0.0],
                         [ 0.5,  0.5, 0.0],
                         [-0.5,  0.5, 0.0]], dtype="float32")
        ncorn = cv2.cornerSubPix(self.gray, imgp, (11,11), (-1,-1), crit)
        ret, rvcs, tvcs, inliers = cv2.solvePnPRansac(objp, ncorn, C, dist)
        return rvcs, tvcs