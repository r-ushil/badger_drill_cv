from typing import Optional
import cv2
import mediapipe as mp

from point_projector import PointProjector
from catching_drill_context import CatchingDrillContext

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingDrillFrameContext():
	__frame: cv2.Mat
	__human_pose_estimator: Optional[mp_pose.Pose]

	'''
		:param frame must be BGR
	'''
	def __init__(self, frame):
		self.__frame = frame
		self.__human_pose_estimator = None
		self.__human_landmarks = None
		self.__frame_effects = []

	def frame_hsv(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

	def frame_grey(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)

	def frame_rgb(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)

	def frame_bgr(self):
		return self.__frame

	def register_human_landmarks(self, human_landmarks):
		self.__human_landmarks = human_landmarks

	def get_human_landmarks(self):
		return self.__human_landmarks

	def add_frame_effect(self, frame_effect):
		self.__frame_effects.append(frame_effect)
	
	def get_frame_effects(self):
		return self.__frame_effects