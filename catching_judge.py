import mediapipe as mp
import numpy as np
import cv2

from numpy import array
from pose_estimator import CameraIntrinsics, PoseEstimator
from typing import Optional

from judge import Judge
from katchet_board import KatchetBoard

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class KatchetDrillFrameContext():
	__frame: cv2.Mat
	__cam_pose_estimator: Optional[PoseEstimator]
	__human_pose_estimator: Optional[mp_pose.Pose]

	__katchet_face_poly: cv2.Mat
	__ball_positions: list

	'''
		:param frame must be BGR
	'''
	def __init__(self, frame):
		self.__frame = frame
		self.__cam_pose_estimator = None
		self.__human_pose_estimator = None
		self.__human_landmarks = None
		self.__katchet_face_poly = None
		self.__ball_positions = []

	def frame_hsv(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

	def frame_grey(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)

	def frame_rgb(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)

	def frame_bgr(self):
		return self.__frame

	def register_cam_pose_estimator(self, cam_pose_estimator: PoseEstimator):
		self.__cam_pose_estimator = cam_pose_estimator

	def get_cam_pose_estimator(self) -> PoseEstimator:
		return self.__cam_pose_estimator

	def register_human_pose_estimator(self, human_pose_estimator: mp_pose.Pose):
		self.__human_pose_estimator = human_pose_estimator

	def get_human_pose_estimator(self) -> mp_pose.Pose:
		return self.__human_pose_estimator

	def register_human_landmarks(self, human_landmarks):
		self.__human_landmarks = human_landmarks

	def get_human_landmarks(self):
		return self.__human_landmarks

	def register_katchet_board_poly(self, katchet_face_poly):
		self.__katchet_face_poly = katchet_face_poly

	def get_katchet_face_poly(self) -> cv2.Mat:
		return self.__katchet_face_poly

	def register_ball_position(self, ball_position):
		self.__ball_positions.append(ball_position)

	def get_ball_positions(self) -> list:
		return self.__ball_positions

class CatchingJudge(Judge):
	__cam_pose_estimator: PoseEstimator
	__cam_intrinsics: CameraIntrinsics

	def __init__(self, input_video_path, cam_intrinsics: CameraIntrinsics):
		super().__init__(input_video_path)

		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			model_complexity=2,    
		)

		self.ball_positions = []

		self.__cam_intrinsics = cam_intrinsics
		self.__cam_pose_estimator = PoseEstimator(cam_intrinsics)

	def process_and_write_video(self):
		for frame in self.get_frames():
			self.process_frame(frame)

	def _resize(self, img):
		return cv2.resize(img, (375, 750))

	def detect_ball(self, frame_context: KatchetDrillFrameContext):
		frame = frame_context.frame_hsv()

		# define range of blue color in HSV (red turns to blue in HSV)
		lower_blue = np.array([160, 160, 100])
		upper_blue = np.array([190, 160, 135])

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(frame, lower_blue, upper_blue)

		# use morphology to remove noise
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=5)

		# find the circle blobs in the mask
		contours, _ = cv2.findContours(
			mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		detected = []

		for _, c in enumerate(contours):
			# get circle area
			area = cv2.contourArea(c)

			# get circle perimeter
			perimeter = cv2.arcLength(c, True)

			# get circlularity
			circularity = 4 * np.pi * (area / (perimeter * perimeter))

			min_circularity = 0.6
			min_area = 30

			(x, y), radius = cv2.minEnclosingCircle(c)

			centre = (int(x), int(y))
			radius = int(radius)

			# add blob information if exceeds thresholds
			if circularity > min_circularity and area > min_area:
				detected.append((area, centre, radius))

		# sort by area
		detected.sort(key=lambda x: x[0], reverse=True)

		# draw the smallest circle
		if len(detected) > 0:
			[ball_position] = detected
			frame_context.register_ball_position(ball_position)

		
	def detect_pose(self, frame_context: KatchetDrillFrameContext):
		# Convert the BGR image to RGB before processing.
		results = self.pose_estimator.process(frame_context.frame_rgb())

		frame_context.register_human_landmarks(results.pose_landmarks)
		frame_context.register_human_pose_estimator(results)

	def katchet_board_detection(self, frame_context: KatchetDrillFrameContext):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)

		frame = frame_context.frame_hsv()
		frame = cv2.GaussianBlur(frame, (9, 9), cv2.BORDER_DEFAULT)

		def h(val):
			return float(val) * (180.0 / 360.0)

		def s(val):
			return float(val) * (255.0 / 100.0)

		def v(val):
			return float(val) * (255.0 / 100.0)

		mask = cv2.inRange(frame,
						   (h(0), s(30), v(80)),
						   (h(30), s(100), v(100)))

		contours, hierarchy = cv2.findContours(
			image=mask,
			mode=cv2.RETR_TREE,
			method=cv2.CHAIN_APPROX_SIMPLE
		)

		contour_lens = [(cv2.arcLength(contour, closed=False), contour)
						for contour in contours]

		# get the longest contour from the list
		contour_lens.sort(key=lambda x: x[0], reverse=True)

		# if no contours were found, return the original frame
		if len(contour_lens) == 0:
			return

		katchet_face_len = contour_lens[0][0]
		katchet_face = contour_lens[0][1]

		epsilon = 0.0125 * katchet_face_len
		katchet_face_poly = cv2.approxPolyDP(
			katchet_face, epsilon=epsilon, closed=True)

		# if polygon is not a quadrilateral
		if not len(katchet_face_poly) == 4:
			return

		frame_context.register_katchet_board_poly(katchet_face)

		katchet_face_pts = np.reshape(katchet_face_poly, (4, 2))

		cam_pose_estimator = PoseEstimator(self.__cam_intrinsics)

		CatchingJudge.__compute_camera_localisation_from_katchet(cam_pose_estimator, katchet_face_pts)

		frame_context.register_cam_pose_estimator(cam_pose_estimator)

	def process_frame(self, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.flip(frame, -1)

		frame_context = KatchetDrillFrameContext(frame)

		self.katchet_board_detection(frame_context)
		self.detect_ball(frame_context)

		# run pose estimation on frame
		self.detect_pose(frame_context)

		output_frame = self.generate_output_frame(frame_context)

		self.write_frame(output_frame)

	def generate_output_frame(self, frame_context: KatchetDrillFrameContext) -> cv2.Mat:
		frame = frame_context.frame_bgr()
		cam_pose_estimator = frame_context.get_cam_pose_estimator()
		katchet_face_poly = frame_context.get_katchet_face_poly()
		ball_positions = frame_context.get_ball_positions()
		human_landmarks = frame_context.get_human_landmarks()

		if cam_pose_estimator is not None:
			CatchingJudge.__render_ground_plane(cam_pose_estimator, frame)

		if katchet_face_poly is not None:
			cv2.drawContours(
				image=frame,
				contours=[katchet_face_poly],
				contourIdx=0,
				color=(0, 0, 255),
				thickness=2,
				lineType=cv2.LINE_AA
			)

		for (area, centre, radius) in ball_positions:
			cv2.circle(frame, centre, radius, (0, 255, 0), cv2.FILLED)

		if human_landmarks is not None:
			mp_drawing.draw_landmarks(frame, human_landmarks, mp_pose.POSE_CONNECTIONS)
		
		return frame

	@staticmethod
	def __compute_camera_localisation_from_katchet(cam_pose_estimator: PoseEstimator, katchet_face):
		katchet_board = KatchetBoard.from_vertices_2d(katchet_face)

		inliers = np.zeros((4, 3), dtype=np.float64)

		return cam_pose_estimator.localise_camera(
			points_3d=katchet_board.get_vertices_3d(),
			points_2d=katchet_board.get_vertices_2d(),
			iterations=500,
			reprojection_err=2.0,
			inliners=inliers,
			confidence=0.95,
		)

	@staticmethod
	def __render_ground_plane(pose_estimator: PoseEstimator, mask):
		for x in range(-4, 6):
			for y in range(-4, 6):
				CatchingJudge.__draw_point(pose_estimator, array([[x], [y], [.0]], dtype=np.float64), mask)

	@staticmethod
	def __draw_point(pose_estimator: PoseEstimator, point: np.ndarray[(3, 1), np.float64], mask):
		pt = pose_estimator.project_3d_to_2d(point).astype('int')
		center = (pt[0], pt[1])
		
		cv2.circle(mask, center, 10, (0, 0, 255), -1)
