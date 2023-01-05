from typing import List
import numpy as np
import mediapipe as mp
import cv2

from numpy import array
from plane import Plane
from point_projector import CameraIntrinsics, PointProjector

from judge import Judge
from katchet_board import KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from frame_effect import FrameEffectType, FrameEffect
from catching_drill_context import CatchingDrillContext
from catching_drill_frame_context import CatchingDrillFrameContext

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingJudge(Judge):
	__cam_pose_estimator: PointProjector

	def __init__(self, input_video_path, cam_intrinsics: CameraIntrinsics):
		super().__init__(input_video_path)

		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			model_complexity=2,    
		)

		self.__cam_intrinsics = cam_intrinsics

	def process_and_write_video(self):
		drill_context = CatchingDrillContext()
		frame_contexts = []

		for frame in self.get_frames():
			frame_contexts.append(self.process_frame(drill_context, frame))
		
		drill_context.generate_heel_2d_positions(self.get_video_dims())
		drill_context.generate_heel_3d_positions()
		drill_context.generate_frame_effects()

		self.write_video(drill_context, frame_contexts)

	def process_frame(self, drill_context: CatchingDrillContext, frame):
		frame = cv2.flip(frame, -1)
		drill_context.frames.append(frame)

		frame_context = CatchingDrillFrameContext(frame)

		katchet_face_pts = self.detect_katchet_board(frame_context)
		drill_context.katchet_faces.append(katchet_face_pts)

		point_projector = None
		if katchet_face_pts is not None:
			point_projector = PointProjector.initialize_from_katchet_face_pts(self.__cam_intrinsics, katchet_face_pts)
		drill_context.point_projectors.append(point_projector)

		ball_position = self.detect_ball(drill_context, frame_context)
		drill_context.ball_positions.append(ball_position)

		# registers pose on frame context, does so using frame and
		drill_context.pose_landmarkss.append(self.detect_pose(frame_context))
		
# 		if katchet_face_pts is not None:
# 			self.detect_human_feet(drill_context, frame_context)

		return frame_context

	def write_video(self, drill_context, frame_contexts: List[CatchingDrillFrameContext]):
		output_frames = self.generate_output_frames(drill_context, frame_contexts)
		for output_frame in output_frames:
			self.write_frame(output_frame)

	def detect_ball(self, drill_context: CatchingDrillContext, frame_context: CatchingDrillFrameContext):
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
			return ball_position
		else:
			return None

	def detect_pose(self, frame_context: CatchingDrillFrameContext):
		return self.pose_estimator.process(frame_context.frame_rgb()).pose_landmarks

	def detect_katchet_board(self, frame_context: CatchingDrillFrameContext):
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
			return None

		katchet_face_len = contour_lens[0][0]
		katchet_face = contour_lens[0][1]

		epsilon = 0.0125 * katchet_face_len
		katchet_face_poly = cv2.approxPolyDP(
			katchet_face, epsilon=epsilon, closed=True)

		# if polygon is not a quadrilateral
		if not len(katchet_face_poly) == 4:
			return None

		frame_context.add_frame_effect(FrameEffect(
			primary_label="Katchet Face Poly",
			frame_effect_type=FrameEffectType.KATCHET_FACE_POLY,
			katchet_face_poly=katchet_face,
			colour=(0, 0, 0),
		))

		katchet_face_pts = np.reshape(katchet_face_poly, (4, 2))

		return katchet_face_pts

	def detect_human_feet(self, drill_context, frame_context: CatchingDrillFrameContext):
		cam_pose_estimator = drill_context.cam_pose_estimators[-1] if len(drill_context.cam_pose_estimators) > 0 else None
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

		x_plane = Plane(
			np.array([0, 0, 0]),
			np.array([1, 0, 0]),
			np.array([1, 0, 1])
		)

		angle_between_planes = trajectory_plane.calculate_angle_with_plane(x_plane)

		circle_initial_point = np.array([2, 0, 0])
		point_rotation_matrix = \
			Plane.get_rotation_matrix_about_point(np.pi / 4, np.array([0, 0, 0]), axis="Z")

		circle_points = []
		current_point = np.append(circle_initial_point, 1)
		for x in range(8):
			circle_points.append(np.delete(current_point, -1))
			current_point = point_rotation_matrix @ current_point 
			
		trajectory_plane_points = trajectory_plane.sample_grid_points(20, 1)
		x_plane_points = x_plane.sample_grid_points(20, 1)

		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		trajectory_plane_points = list(filter(plane_points_to_print, trajectory_plane_points))
		x_plane_points = list(filter(plane_points_to_print, x_plane_points))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_MULTIPLE,
			primary_label="Trajectory plane points",
			points_multiple=trajectory_plane_points,
			colour=(0, 0, 255),
			show_label=False
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_MULTIPLE,
			primary_label="X plane points",
			points_multiple=x_plane_points,
			colour=(0, 0, 255),
			show_label=False
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.TEXT,
			primary_label="Angle between planes",
			display_label=f"t = {(angle_between_planes / (2 * np.pi)) * 360}",
			show_label=True
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.TEXT,
			primary_label="Angle between planes",
			display_label=f"180 - t = {180 - ((angle_between_planes / (2 * np.pi)) * 360)}",
			show_label=True
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINT_SINGLE,
			point_single=left_heel_world,
			primary_label="Left Heel",
			display_label=FrameEffect.generate_point_string(left_heel_world),
			show_label=True,
			colour=(255, 0, 0)
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINT_SINGLE,
			point_single=right_heel_world,
			primary_label="Right Heel",
			display_label=FrameEffect.generate_point_string(right_heel_world),
			show_label=True,
			colour=(255, 0, 0)
		))

		frame_context.add_frame_effect(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_MULTIPLE,
			primary_label="Circle points",
			points_multiple=circle_points,
			colour=(0, 255, 0),
			show_label=False
		))

	def generate_output_frames(self, drill_context: CatchingDrillContext, frame_context: CatchingDrillFrameContext) -> cv2.Mat:
		# frame = frame_context.frame_bgr()

		augmented_frames = []
		for frame, point_projector, frame_effects in zip(drill_context.frames, drill_context.point_projectors, drill_context.frame_effectss):

			# ball_positions = drill_context.ball_positions
			# cam_pose_estimator = drill_context.cam_pose_estimators[-1]

			# human_landmarks = frame_context.get_human_landmarks()

			# for (area, centre, radius) in ball_positions:
			# 	cv2.circle(frame, centre, radius, (0, 255, 0), cv2.FILLED)

			# if cam_pose_estimator is not None:
			# 	CatchingJudge.__render_ground_plane(cam_pose_estimator, frame)

			# if human_landmarks is not None:
			# 	mp_drawing.draw_landmarks(frame, human_landmarks, mp_pose.POSE_CONNECTIONS)
		
			# frame_context.add_frame_effect(FrameEffect(
			# 	frame_effect_type=FrameEffectType.POINTS_MULTIPLE,
			# 	primary_label="Katchet board points",
			# 	points_multiple=np.array([KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R]),
			# 	colour=(0, 0, 255),
			# 	show_label=True
			# ))

			if point_projector is not None:
				label_counter = 1
				for effect in frame_effects:
					match effect.frame_effect_type:
						case FrameEffectType.POINTS_MULTIPLE:
							for point in effect.points_multiple:
								CatchingJudge.__label_point(point_projector, point, frame, None, effect.show_label, colour=effect.colour)
						case FrameEffectType.POINT_SINGLE:
							CatchingJudge.__label_point(point_projector, effect.point_single, frame, effect.display_label, show_label=effect.show_label, colour=effect.colour)
						case FrameEffectType.KATCHET_FACE_POLY:
							cv2.drawContours(
								image=frame,
								contours=[effect.katchet_face_poly],
								contourIdx=0,
								color=effect.colour,
								thickness=2,
								lineType=cv2.LINE_AA
							)
						case FrameEffectType.TEXT:
							if effect.show_label:
								cv2.putText(
									frame,
									effect.display_label,
									(20, label_counter * 40),
									fontFace=cv2.FONT_HERSHEY_SIMPLEX,
									fontScale=1,
									color=(0, 0, 0),
									thickness=2,
									lineType=cv2.LINE_AA,
								)
								label_counter += 1

			augmented_frames.append(frame)

		return augmented_frames

	@staticmethod
	def jjj__render_ground_plane(pose_estimator: PointProjector, mask):
		for x in range(-4, 6):
			for y in range(-4, 6):
				CatchingJudge.__draw_point(pose_estimator, array([[x], [y], [.0]], dtype=np.float64), mask)

	@staticmethod
	def __draw_point(pose_estimator: PointProjector, point: np.ndarray[(3, 1), np.float64], mask):
		pt = pose_estimator.project_3d_to_2d(point).astype('int')
		center = (pt[0], pt[1])
		
		cv2.circle(mask, center, 2, (255, 0, 0), -1)

	@staticmethod
	def __label_point(pose_estimator: PointProjector, point_3d: np.ndarray[(3, 1), np.float64], mask, label: str, show_label = True, colour = (255, 0, 0)):
		wx, wy, wz = point_3d[0], point_3d[1], point_3d[2]

		point_2d = pose_estimator.project_3d_to_2d(point_3d).astype('int')
		sx, sy = point_2d[0], point_2d[1]

		cv2.circle(mask, (sx, sy), 10, colour, -1)
		if show_label:
			cv2.putText(
				mask,
				label if label is not None else FrameEffect.generate_point_string(point_3d),
				(sx, sy),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(255, 255, 255),
				thickness=2,
				lineType=cv2.LINE_AA,
			)
