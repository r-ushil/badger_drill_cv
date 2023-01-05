import cv2

class AugmentedFrame():
	__frame: cv2.Mat

	'''
		:param frame must be BGR
	'''
	def __init__(self, frame):
		self.__frame = frame

	def frame_hsv(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

	def frame_grey(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)

	def frame_rgb(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)

	def frame_bgr(self):
		return self.__frame