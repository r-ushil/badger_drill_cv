from typing import Optional
from augmented_frame import AugmentedFrame
from cv2 import arcLength, \
	contourArea, \
	minEnclosingCircle, \
	findContours, \
	inRange, \
	morphologyEx, \
	CHAIN_APPROX_SIMPLE, \
	MORPH_DILATE, \
	RETR_EXTERNAL
from numpy import array, ones, uint8, pi

class BallDetector():
	ball_positions: list

	def __init__(self) -> None:
		self.ball_positions = []

	def process(self, augmented_frame: AugmentedFrame):
		frame = augmented_frame.frame_hsv()

		# define range of blue color in HSV (red turns to blue in HSV)
		lower_blue = array([160, 160, 90])
		upper_blue = array([200, 160, 135])

		# Threshold the HSV image to get only blue colors
		mask = inRange(frame, lower_blue, upper_blue)

		# use morphology to remove noise
		kernel = ones((3, 3), uint8)
		mask = morphologyEx(mask, MORPH_DILATE, kernel, iterations=5)

		# find the circle blobs in the mask
		contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

		detected = []

		for _, c in enumerate(contours):
			# get circle area
			area = contourArea(c)

			# get circle perimeter
			perimeter = arcLength(c, True)

			# get circlularity
			circularity = 4 * pi * (area / (perimeter * perimeter))

			min_circularity = 0.6
			min_area = 30

			(x, y), radius = minEnclosingCircle(c)

			centre = (int(x), int(y))
			radius = int(radius)

			# add blob information if exceeds thresholds
			if circularity > min_circularity and area > min_area:
				detected.append((area, centre, radius))

		# sort by area
		detected.sort(key=lambda x: x[0], reverse=True)

		# draw the smallest circle
		if len(detected) > 0:
			(_, center, _) = detected[0]
			self.ball_positions.append(center)
		else:
			self.ball_positions.append(None)
