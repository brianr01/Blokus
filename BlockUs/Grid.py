import cv2
import numpy as np

class Grid:
	def pluckBoardSquares(self, image):
		unit = 28.35
		subSelectMargin = 7
		subSelectUnit =  int(unit - subSelectMargin * 2)
		newBoardWidth = int(20 * subSelectUnit)
		newBoardHeight = int(newBoardWidth)
		newBoard = np.zeros((newBoardHeight,newBoardWidth,3), np.uint8)
		for i in range(0, 20):
			for j in range(0, 20):
				originX = int(i * unit)
				originY = int(j * unit)


				x0 = originX + subSelectMargin
				y0 = originY + subSelectMargin

				x1 = int(originX + (unit - subSelectMargin))
				y1 = int(originY + (unit - subSelectMargin))

				point1 = (x0, y0)
				point2 = (x1, y1)

				newBoard[i * subSelectUnit: i * subSelectUnit + subSelectUnit, j * subSelectUnit: j * subSelectUnit + subSelectUnit] = image[x0:x1, y0:y1]

		return newBoard


