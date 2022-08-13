
import cv2
import numpy as np
from FishEye import FishEye 
from FishEye import DistortionCalibration
from Aruco import Aruco
import glob
import math
from scipy.spatial import distance as dist
from Perspective import Perspective
from Grid import Grid

debug = True
stopOnDebug = False and debug


def load_images(path, extension):
        image_links = glob.glob(path + "\\*" + extension)

        loaded_images = []
        for image_link in image_links:
            loaded_images.append(cv2.imread(image_link))

        return loaded_images

fishEye = FishEye()
#images = load_images("C:\\Users\\brian\\Pictures\\Camera Roll\\calibration", ".jpg")
#calibrationOBJ = fishEye.get_calibration(images)
#fishEye.save_calibration(calibrationOBJ)
calibrationOBJ = fishEye.load_calibration()

referenceImage = cv2.imread("C:\\Users\\brian\\Pictures\\Camera Roll\\WIN_20220810_00_43_51_Pro.jpg")

perspective = Perspective()

aruco = Aruco()

#centers = aruco.getCenters(undistortedImage)

#if not len(centers) == 4:
	#	undistortedImage = aruco.drawCenters(undistortedImage, centers)
	#	cv2.imshow("aruco", undistortedImage)
	#	cv2.waitKey(0)
	#	exit()

#perspectiveTransformMatrix = perspective.getTransformMatrix(centers)
#perspective.saveTransform(perspectiveTransformMatrix)
perspectiveTransformMatrix = perspective.loadTransform()

# Dynamic variable default set close to expected number.
unit = 28

def cropImageToBoard(image, centers, unit):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    center = (sum(x) / len(points), sum(y) / len(points))
    image = image[10 * unit - center: 10 * unit - center, 20 * unit - center: 20 * unit - center]

    return image

def order_points(pts):
		# sort the points based on their x-coordinates
		xSorted = pts[np.argsort(pts[:, 0]), :]
		# grab the left-most and right-most points from the sorted
		# x-roodinate points
		leftMost = xSorted[:2, :]
		rightMost = xSorted[2:, :]
		# now, sort the left-most coordinates according to their
		# y-coordinates so we can grab the top-left and bottom-left
		# points, respectively
		leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
		(tl, bl) = leftMost
		# now that we have the top-left coordinate, use it as an
		# anchor to calculate the Euclidean distance between the
		# top-left and right-most points; by the Pythagorean
		# theorem, the point with the largest distance will be
		# our bottom-right point
		D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
		(br, tr) = rightMost[np.argsort(D)[::-1], :]
		# return the coordinates in top-left, top-right,
		# bottom-right, and bottom-left order
		return np.array([tl, tr, br, bl], dtype="float32")

def final(originalImage):
    global perspectiveTransformMatrix, unit
    if debug:
        cv2.imshow('original', originalImage)
        if stopOnDebug:
            cv2.waitKey(0)

    undistortedImage = fishEye.undistort_image(originalImage, calibrationOBJ)

    if debug:
        cv2.imshow('undistorted', undistortedImage);
        if stopOnDebug:
            cv2.waitKey(0)

    #centers = aruco.getCenters(undistortedImage)


    centers = aruco.getCenters(undistortedImage)
    if debug:
        cv2.imshow('aruco', aruco.drawCenters(undistortedImage, centers))
        if stopOnDebug:
            cv2.waitKey(0)

    unitSquare = aruco.getSquareUnit(undistortedImage)

    print(unit)

    if (len(centers) == 4):
        perspectiveTransformMatrix = perspective.getTransformMatrix(centers)
    else:
        print(['aruo not found.', len(centers)])

    orthoImage = perspective.transform(undistortedImage, perspectiveTransformMatrix)

    if debug:
        cv2.imshow('ortho', orthoImage)
        if stopOnDebug:
            cv2.waitKey(0)

    #centers2 = aruco.getCenters(orthoImage)
    #if debug:
    #    cv2.imshow('aruco2', aruco.drawCenters(orthoImage, centers2))
    #    if stopOnDebug:
    #        cv2.waitKey(0)

    #unitSquare = aruco.getSquareUnit(orthoImage)
    #squares = aruco.getCentersForSquares(orthoImage)
    #points = order_points(np.float32(squares))
    if unitSquare:
        unit = unitSquare

    #cv2.imshow('copped', cropImageToBoard(orthoImage, squares, unit))


    grid = Grid()
    boardImage = grid.pluckBoardSquares(orthoImage)

    if debug:
        cv2.imshow('boardImage', boardImage)
        if stopOnDebug:
            cv2.waitKey(0)

    return boardImage

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('open cap')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    output = final(frame)
    # Display the resulting frame
    cv2.imshow('frame', output)
    if cv2.waitKey(1) == ord('q'):
        break



