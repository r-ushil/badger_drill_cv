import mediapipe as mp
import numpy as np
import cv2

from numpy import array
from pose_estimator import CameraIntrinsics, PoseEstimator

from judge import Judge
from katchet_board import KatchetBoard

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

	def detect_ball(self, frame):
		# convert to HSV
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

		frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

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
			self.ball_positions.append(detected[0])

		for (area, centre, radius) in self.ball_positions:
			cv2.circle(frame, centre, radius, (0, 255, 0), cv2.FILLED)

		# UNCOMMENT TO SHOW MASK FOR DEBUGGING
		# cv2.imshow('frame', self._resize(frame))
		# cv2.waitKey(1)

		return frame

		
	def detect_pose(self, frame):
		# Convert the BGR image to RGB before processing.
		results = self.pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		mp_drawing.draw_landmarks(
			frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# UNCOMMENT TO SHOW POSE DETECTION FOR DEBUGGING
		# cv2.imshow('MediaPipe Pose', self._resize(frame))
		# cv2.waitKey(1)

		return frame

	def katchet_board_detection(self, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

		mask = np.zeros(shape=mask.shape, dtype=np.uint8)
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

		contour_lens = [(cv2.arcLength(contour, closed=False), contour)
						for contour in contours]

		# get the longest contour from the list
		contour_lens.sort(key=lambda x: x[0], reverse=True)

		# if no contours were found, return the original frame
		if len(contour_lens) == 0:
			# Detected no contours
			mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
			return mask

		katchet_face_len = contour_lens[0][0]
		katchet_face = contour_lens[0][1]

		epsilon = 0.0125 * katchet_face_len
		katchet_face_poly = cv2.approxPolyDP(
			katchet_face, epsilon=epsilon, closed=True)

		if not len(katchet_face_poly) == 4:
			# Failed to detect a quadrilateral
			mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
			return mask

		cv2.drawContours(
			image=mask,
			contours=[katchet_face_poly],
			contourIdx=0,
			color=(0, 0, 255),
			thickness=2,
			lineType=cv2.LINE_AA
		)

		katchet_face_pts = np.reshape(katchet_face_poly, (4, 2))

		self.__compute_camera_localisation_from_katchet(katchet_face_pts)
		self.__render_ground_plane(mask)

		mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
		return mask

	def process_frame(self, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.flip(frame, -1)

		board_detected = self.katchet_board_detection(frame)
		ball_detected = self.detect_ball(frame)

		# run pose estimation on frame
		pose_detected = self.detect_pose(ball_detected)

		output_frame = cv2.bitwise_or(board_detected, pose_detected)

		self.write_frame(output_frame)

	def __compute_camera_localisation_from_katchet(self, katchet_face):
		katchet_board = KatchetBoard.from_vertices_2d(katchet_face)

		inliers = np.zeros((4, 3), dtype=np.float64)

		return self.__cam_pose_estimator.localise_camera(
			points_3d=katchet_board.get_vertices_3d(),
			points_2d=katchet_board.get_vertices_2d(),
			iterations=500,
			reprojection_err=2.0,
			inliners=inliers,
			confidence=0.95,
		)

	def __render_ground_plane(self, mask):
		for x in range(-4, 6):
			for y in range(-4, 6):
				self.__draw_point(array([[x], [y], [.0]], dtype=np.float64), mask)

	def __draw_point(self, point: np.ndarray[(3, 1), np.float64], mask):
		pt = self.__cam_pose_estimator.project_3d_to_2d(point).astype('int')
		center = (pt[0], pt[1])
		
		cv2.circle(mask, center, 10, (0, 0, 255), -1)
