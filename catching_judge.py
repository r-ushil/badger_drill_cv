import mediapipe as mp
import numpy as np
import cv2

from numpy import array, around
from plane import Plane
from pose_estimator import CameraIntrinsics, PoseEstimator
from typing import Optional

from judge import Judge
from katchet_board import KatchetBoard, KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from frame_effect import FrameEffectType, FrameEffect

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class KatchetDrillContext():
	__ball_positions: list

	def __init__(self) -> None:
		self.__ball_positions = []

	def register_ball_position(self, ball_position):
		self.__ball_positions.append(ball_position)

	def get_ball_positions(self) -> list:
		return self.__ball_positions

class KatchetDrillFrameContext():
	__frame: cv2.Mat
	__cam_pose_estimator: Optional[PoseEstimator]
	__human_pose_estimator: Optional[mp_pose.Pose]

	__katchet_face_poly: cv2.Mat

	'''
		:param frame must be BGR
	'''
	def __init__(self, drill_context: KatchetDrillContext, frame):
		self.__drill_context = drill_context

		self.__frame = frame
		self.__cam_pose_estimator = None
		self.__human_pose_estimator = None
		self.__human_landmarks = None
		self.__katchet_face_poly = None
		self.__trajectory_plane_points = None
		self.__frame_effects = []

	def frame_hsv(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2HSV)

	def frame_grey(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2GRAY)

	def frame_rgb(self):
		return cv2.cvtColor(self.__frame, cv2.COLOR_BGR2RGB)

	def frame_bgr(self):
		return self.__frame

	def drill_context(self) -> KatchetDrillContext:
		return self.__drill_context

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

	def register_trajectory_plane_points(self, trajectory_plane_points):
		self.__trajectory_plane_points = trajectory_plane_points

	def get_trajectory_plane_points(self):
		return self.__trajectory_plane_points

	def add_frame_effect(self, frame_effect):
		self.__frame_effects.append(frame_effect)
	
	def get_frame_effects(self):
		return self.__frame_effects

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
		context = KatchetDrillContext()

		for frame in self.get_frames():
			self.process_frame(context, frame)

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
			ball_position = detected[0]
			drill_context = frame_context.drill_context()
			drill_context.register_ball_position(ball_position)

		
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
						   (h(40), s(100), v(100)))

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

	def localise_human_feet(self, frame_context: KatchetDrillFrameContext):
		cam_pose_estimator = frame_context.get_cam_pose_estimator()
		pose_landmarks = frame_context.get_human_landmarks()

		if cam_pose_estimator is None or pose_landmarks is None:
			return

		left_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
		right_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

		vid_w, vid_h = self.get_video_dims()
		sx_left_foot = left_heel.x * vid_w
		sy_left_foot = left_heel.y * vid_h
		sx_right_foot = right_heel.x * vid_w
		sy_right_foot = right_heel.y * vid_h

		left_heel_screen_coordinates = array([sx_left_foot, sy_left_foot])
		right_heel_screen_coordinates = array([sx_right_foot, sy_right_foot])

		left_heel_world = cam_pose_estimator.project_2d_to_3d(left_heel_screen_coordinates, Z=0)
		right_heel_world = cam_pose_estimator.project_2d_to_3d(right_heel_screen_coordinates, Z=0)

		trajectory_plane = Plane(
			np.array([0, 0.5, 0]),
			np.array(left_heel_world.reshape(3,)),
			np.array([0, 0.5, -1])
		)

		circle_initial_point = np.array([2, 0, 0])
		point_rotation_matrix = \
			Plane.get_rotation_matrix_about_point(np.pi / 4, np.array([0, 0, 0]), axis="Z")

		circle_points = []
		current_point = np.append(circle_initial_point, 1)
		for x in range(8):
			circle_points.append(np.delete(current_point, -1))
			current_point = point_rotation_matrix @ current_point 
			

		trajectory_plane_points = trajectory_plane.sample_grid_points(20, 1)
		frame_context.register_trajectory_plane_points(trajectory_plane_points)

		# TODO: Make take colour into account
		# TODO: Make optionally print label
		# TODO: Add mandatory searchable label, optional display label
		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINT_SINGLE,
			point_single=left_heel_world,
			label="Left Heel",
			show_label=True,
			colour=(255, 0, 0)
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINT_SINGLE,
			point_single=right_heel_world,
			label="Right Heel",
			show_label=True,
			colour=(255, 0, 0)
		))


		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_MULTIPLE,
			points_multiple=circle_points,
			colour=(0, 255, 0)
		))

	def process_frame(self, context: KatchetDrillContext, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.flip(frame, -1)

		frame_context = KatchetDrillFrameContext(context, frame)

		self.katchet_board_detection(frame_context)
		self.detect_ball(frame_context)

		# run pose estimation on frame
		self.detect_pose(frame_context)

		self.localise_human_feet(frame_context)

		output_frame = self.generate_output_frame(frame_context)

		self.write_frame(output_frame)

	def generate_output_frame(self, frame_context: KatchetDrillFrameContext) -> cv2.Mat:
		drill_context = frame_context.drill_context()

		frame = frame_context.frame_bgr()

		ball_positions = drill_context.get_ball_positions()

		cam_pose_estimator = frame_context.get_cam_pose_estimator()
		katchet_face_poly = frame_context.get_katchet_face_poly()
		human_landmarks = frame_context.get_human_landmarks()
		trajectory_plane_points = frame_context.get_trajectory_plane_points()
		frame_effects = frame_context.get_frame_effects()

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

		for effect in frame_effects:
			match effect.frame_effect_type:
				case FrameEffectType.POINTS_MULTIPLE:
					for point in effect.points_multiple:
						CatchingJudge.__label_point(cam_pose_estimator, point, frame, "")
				case FrameEffectType.POINT_SINGLE:
					CatchingJudge.__label_point(cam_pose_estimator, effect.point_single, frame, effect.label)


		if trajectory_plane_points is not None:
			for p in trajectory_plane_points:
				(x, y, z) = p
				if z <= 0 and z > -0.8 and x >= -2:
					CatchingJudge.__label_point(cam_pose_estimator, p.reshape((3, 1)), frame, "", False)

		if cam_pose_estimator is not None:
			CatchingJudge.__label_point(cam_pose_estimator, KATCHET_BOX_BOT_L, frame, "KB:BL")
			CatchingJudge.__label_point(cam_pose_estimator, KATCHET_BOX_BOT_R, frame, "KB:BR")
			CatchingJudge.__label_point(cam_pose_estimator, KATCHET_BOX_TOP_L, frame, "KB:TL")
			CatchingJudge.__label_point(cam_pose_estimator, KATCHET_BOX_TOP_R, frame, "KB:TR")
		
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
		
		cv2.circle(mask, center, 2, (255, 0, 0), -1)

	@staticmethod
	def __label_point(pose_estimator: PoseEstimator, point_3d: np.ndarray[(3, 1), np.float64], mask, label: str, show_label = True):
		wx, wy, wz = point_3d[0], point_3d[1], point_3d[2]

		point_2d = pose_estimator.project_3d_to_2d(point_3d).astype('int')
		sx, sy = point_2d[0], point_2d[1]

		cv2.circle(mask, (sx, sy), 10, (255, 0, 0), -1)
		if show_label:
			cv2.putText(
				mask,
				f"({around(wx, 2)}, {around(wy, 2)}, {wz})", (sx, sy),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(255, 255, 255),
				thickness=2,
				lineType=cv2.LINE_AA,
			)
