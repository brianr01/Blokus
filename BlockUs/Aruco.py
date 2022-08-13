import cv2
import numpy as np
import math

class Aruco:
	def detectArucos(self, image):
		arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
		arucoParams = cv2.aruco.DetectorParameters_create()
		(squares, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)

		return squares

	def getCentersForSquares(self, squares):
		centers = []

		for square in squares:
			points = np.array(square, np.int32)
			points = points.reshape((-1, 1, 2))

			x = [p[0][0] for p in points]
			y = [p[0][1] for p in points]
			center = (sum(x) / len(points), sum(y) / len(points))
			centers.append(center)

		return centers

	def drawCenters(self, image, centers):
		for center in centers:
			image = cv2.circle(image, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
			image = cv2.circle(image, (int(center[0]), int(center[1])), 3, (255, 0, 0), -1)

		return image

	def getCenters(self, image):
		squares = self.detectArucos(image)
		centers = self.getCentersForSquares(squares)

		return centers

	def getSquareUnit(self, image):
		squares = self.detectArucos(image)

		distances = []
		for square in squares:
			points = np.array(square, np.int32)
			points = points.reshape((-1, 1, 2))
			pointCount = len(points)
			for i in range(0, len(points)):
				point0 = points[i][0]
				point1 = points[(i + 1) % pointCount][0]
				distance = self.calculateDistance(point0[0], point0[1], point1[0], point1[1])
				distances.append(distance)

		if not len(distances):
			return None

		return sum(distances) / len(distances)

	def calculateDistance(self, x1,y1,x2,y2):
		dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
		return dist



