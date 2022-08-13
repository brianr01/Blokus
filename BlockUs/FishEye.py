import numpy as np
import cv2 as cv
import glob
import pickle

class DistortionCalibration:
    def __init__(self, ret, cameraMatrix, dist, rvecs, tvecs):
        self.ret = ret
        self.cameraMatrix = cameraMatrix
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs


class FishEye:
    def __init__(self):
        self.chessboard_size = (10, 7)
        self.frame_size = (1920, 1080)

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,2)    

    def get_calibration(self, images):
        obj_points = []
        img_points = []

        for image in images:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, self.chessboard_size, None)

            if ret == True:
                obj_points.append(self.objp)
                corners2=cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                img_points.append(corners)

        ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, self.frame_size, None, None)
        calibrationObject = DistortionCalibration(ret, cameraMatrix, dist, rvecs, tvecs)

        return calibrationObject

    def save_calibration(self, calibration):
        import pickle

        # Store data (serialize)
        with open('fishEyeCalibration.pickle', 'wb') as handle:
            pickle.dump(calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_calibration(self):
        with open('fishEyeCalibration.pickle', 'rb') as handle:
            unserialized_data = pickle.load(handle)

        return unserialized_data

    def undistort_image(self, image, calibrationObject):
        h, w = image.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(calibrationObject.cameraMatrix, calibrationObject.dist, (w,h), 1, (w,h))

        mapx, mapy = cv.initUndistortRectifyMap(calibrationObject.cameraMatrix, calibrationObject.dist, None, newCameraMatrix, (w, h), 5)
        dst = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst




