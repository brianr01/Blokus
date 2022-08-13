import pickle
import cv2
import numpy as np
from scipy.spatial import distance as dist

class Perspective:
	def order_points(self, pts):
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

	def getTransformMatrix(self, centers):
		points = self.order_points(np.float32(centers))

		#boardSize = [20,20]
		#base = 55
		#far = 512
		#unit = base-far/16

		#x = [p[0] for p in points]
		#y = [p[1] for p in points]
		#center = (sum(x) / len(points), sum(y) / len(points))

		#halfBoardWith = boardSize[0] / 2
		#halfBoardHeight = boardSize[1] / 2

		#lowX = (unit * halfBoardWith) - center[0]
		#highX = (unit * halfBoardWith * 2) - center[0] - lowX
		#lowY = (unit * halfBoardHeight) - center[1]
		#highY = (unit * halfBoardHeight * 2) -  center[1] - lowY
		#highY =(unit * halfBoardHeight * 2) -  center[1]
		#low = 200
		#high = 500
		

		boundingBox = self.boundingBox(points)
		lowX = boundingBox[0][0]
		lowY = boundingBox[0][1]
		highX = boundingBox[1][0]
		highY = boundingBox[1][1]

		endPoints = np.float32([[lowX, lowY], [highX, lowY], [highX, highY], [lowX, highY]])

		matrix = cv2.getPerspectiveTransform(points, endPoints)

		return matrix

	def saveTransform(self, matrix):
        # Store data (serialize)
		with open('perspectiveTransformMatrix.pickle', 'wb') as handle:
			pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def loadTransform(self):
		with open('perspectiveTransformMatrix.pickle', 'rb') as handle:
			unserialized_data = pickle.load(handle)

		return unserialized_data

	def transform(self, image, matrix):
		base = 55
		far = 512
		unit = base-far/16
		image = cv2.warpPerspective(image, matrix, (1920, 1080))

		#image = image[:base+far, :base+far]

		return image

	def calculateDistance(self, x1,y1,x2,y2):
		dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
		return dist

	
	def boundingBox(self, points):
		x_coordinates, y_coordinates = zip(*points)

		return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]



