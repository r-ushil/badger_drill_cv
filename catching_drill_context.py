import mediapipe as mp
import numpy as np

from copy import copy

from ball_detector import \
	BallDetector, \
	CriticalBallPointDetector, \
	CriticalPointType
from catching_drill_results import CatchingDrillResults
from frame_effect import FrameEffect, FrameEffectType
from katchet_board import KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from plane import Plane
from point_projector import PointProjector

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class CatchingDrillContext():
	ball_2d_positions: list
	ball_detector: BallDetector

	def __init__(self, fps) -> None:
		# Exist per frame
		self.fps = fps
		self.frames = []
		self.katchet_faces = []
		self.point_projectors = []
		self.pose_landmarkss = []
		self.ball_2d_filtered_positions = []
		self.ball_detector = BallDetector()

		self.left_heel_2d_positions = []
		self.right_heel_2d_positions = []

		self.left_hand_2d_positions = []
		self.right_hand_2d_positions = []

		self.left_heel_3d_positions = []
		self.right_heel_3d_positions = []

		self.ball_3d_positions = []
		self.ball_decomposed_positions = []

		# Remain the same throughout the drill
		self.x_plane_fixed = None
		self.ground_plane_fixed = None
		self.circle_points_fixed = None
		self.trajectory_plane_fixed = None
		self.angle_between_planes_fixed = None
		self.intersection_point_of_planes_fixed = None
		self.first_trajectory_change_2d_position = None
		self.last_trajectory_change_2d_position = None

		self.frame_effectss = []

	def interpolate_missing_data(self):
		# TODO: interpolate missing Katchetboard coordinates
		from pandas import DataFrame
		from numpy import array, nan, reshape

		measured_katchet_faces = array([
			face.flatten() if face is not None else array([nan, nan, nan, nan, nan, nan, nan, nan]) for face in self.katchet_faces
		], dtype='float64')

		measured_katchet_faces_df = DataFrame(measured_katchet_faces)
		interpol_katchet_faces_df = measured_katchet_faces_df.interpolate(
			method="linear", limit_direction="both")

		self.katchet_faces = reshape(
			interpol_katchet_faces_df.to_numpy(), (-1, 4, 2)).astype(int)
		self.ball_detector.interpolate_ball_positions()

	def generate_augmented_data(self, video_dims, cam_intrinsics):
		self.generate_point_projectors(cam_intrinsics)
		self.generate_hand_2d_positions(video_dims)

		bounce_critical_pt, catch_critical_pt = self.generate_ball_critical_points()
		ball_pts = self.ball_detector.get_ball_positions()

		if bounce_critical_pt is None or catch_critical_pt is None:
			raise "Cannot find bounce or catch points!"

		# replace 89 with catch_pt.get_frame_num()
		# replace this with catch_pt.get_position_2d()

		# TODO: Fix the critical point detector
		# catch_frame = catch_critical_pt.get_frame_num()
		# (catch_x, catch_y) = catch_critical_pt.get_position_2d()
		self.catch_frame_index = 89 # Good for 809 / 002
		self.bounce_frame_index = bounce_critical_pt.get_frame_num()
		(catch_x, catch_y, _) = ball_pts[self.catch_frame_index]

		self.bounce_2d_position = bounce_critical_pt.get_position_2d()
		self.catch_2d_position = (catch_x, catch_y)

		self.ball_2d_filtered_positions = \
			[ball_pt if (frame_index > self.bounce_frame_index and frame_index < self.catch_frame_index) else None
			  for (frame_index, ball_pt) in enumerate(ball_pts)]

		# (self.ball_2d_filtered_positions,
		#  self.first_trajectory_change_2d_position,
		#  self.last_trajectory_change_2d_position) = \
		# 	CatchingDrillContext.filter_ball_2d_positions(self.ball_detector.get_ball_positions())

		self.left_heel_2d_positions, self.right_heel_2d_positions = \
			CatchingDrillContext.generate_heel_2d_positions(
				video_dims, self.pose_landmarkss)

		self.left_heel_3d_positions, self.right_heel_3d_positions = \
			CatchingDrillContext.generate_heel_3d_positions(
				self.left_heel_2d_positions,
				self.right_heel_2d_positions,
				self.point_projectors
			)

		# self.first_trajectory_change_3d_position = \
		# 	CatchingDrillContext.localise_first_trajectory_change_position(
		# 		self.ground_plane_fixed,
		# 		self.first_trajectory_change_2d_position
		# 	)

		self.trajectory_plane_fixed = CatchingDrillContext.generate_trajectory_plane()
		self.x_plane_fixed = CatchingDrillContext.generate_x_plane()
		self.ground_plane_fixed = CatchingDrillContext.generate_ground_plane()

		self.angle_between_planes_fixed = \
			CatchingDrillContext.generate_angle_between_planes(
				self.x_plane_fixed, self.trajectory_plane_fixed)
		self.intersection_point_of_planes_fixed = \
			CatchingDrillContext.generate_intersection_point_of_planes(
				self.x_plane_fixed, self.trajectory_plane_fixed)

		self.ball_3d_positions = CatchingDrillContext.generate_ball_3d_positions(
			[(p[0], p[1]) if p is not None else None for p in self.ball_2d_filtered_positions],
			self.trajectory_plane_fixed,
			self.point_projectors
		)

		self.ball_decomposed_positions = CatchingDrillContext.decompose_ball_3d_positions(
			self.ball_3d_positions,
			self.trajectory_plane_fixed
		)

		from numpy import array, average, gradient
		from numpy.linalg import norm

		ball_positions = array([[pos[0, 0], pos[1, 0], 0]
							   for pos in self.ball_3d_positions if pos is not None])
		[ball_deltas, _] = gradient(ball_positions)

		ball_velocities = ball_deltas * float(self.fps)
		ball_speeds = norm(ball_velocities, axis=1)

		self.ball_velocity_average = average(ball_velocities, axis=0)
		self.ball_speed_average = average(ball_speeds)

		self.ball_displacement_max_height = max([-pos[2, 0] for pos in self.ball_3d_positions if pos is not None])

		# self.generate_circle_points()

	def generate_frame_effects(self):
		ball_2d_positions_so_far = []
		ball_3d_positions_so_far = []

		for (frame_num, (frame,
						 katchet_face,
						 left_heel_3d,
						 right_heel_3d,
						 pose_landmarks,
						 ball_2d_position,
						 ball_3d_position)) in enumerate(zip(
							 self.frames,
							 self.katchet_faces,
							 self.left_heel_3d_positions,
							 self.right_heel_3d_positions,
							 self.pose_landmarkss,
							 self.ball_2d_filtered_positions,
							 self.ball_3d_positions,
						 )):

			frame_effects = []

			# TODO: Add frame effect for pose landmarks rather than directly
			# annotating the frame
			CatchingDrillContext.add_pose_landmarks_frame_effect(
				frame, pose_landmarks)
			# CatchingDrillContext.add_ball_2d_positions_frame_effect(frame_effects, ball_2d_position, ball_2d_positions_so_far)
			CatchingDrillContext.add_ball_3d_positions_frame_effect(
				frame_effects, ball_3d_position, ball_3d_positions_so_far)
			CatchingDrillContext.add_trajectory_change_2d_positions_frame_effect(
				frame_effects,
				self.first_trajectory_change_2d_position,
				self.last_trajectory_change_2d_position
			)
			CatchingDrillContext.add_katchet_face_frame_effect(
				frame_effects, katchet_face)
			CatchingDrillContext.add_katchet_board_points_frame_effect(
				frame_effects)

			CatchingDrillContext.add_left_heel_3d_frame_effect(
				frame_effects, left_heel_3d)
			# CatchingDrillContext.add_right_heel_3d_frame_effect(frame_effects, right_heel_3d)

			CatchingDrillContext.add_trajectory_plane_frame_effect(
				frame_effects, self.trajectory_plane_fixed)
			CatchingDrillContext.add_x_plane_frame_effect(
				frame_effects, self.x_plane_fixed)
			CatchingDrillContext.add_ground_plane_frame_effect(
				frame_effects, self.ground_plane_fixed)

			CatchingDrillContext.add_angle_printing_frame_effect(
				frame_effects, self.angle_between_planes_fixed)
			CatchingDrillContext.add_intersection_point_frame_effect(
				frame_effects, self.intersection_point_of_planes_fixed)
			# CatchingDrillContext.add_circle_frame_effect(frame_effects, self.circle_points_fixed)

			CatchingDrillContext.add_frame_number(frame_effects, frame_num)
			CatchingDrillContext.add_ball_average_speed_frame_effect(
				frame_effects, self.ball_speed_average)

			CatchingDrillContext.ball_displacement_max_height_frame_effect(
				frame_effects, self.ball_displacement_max_height)

			# CatchingDrillContext.add_ball_velocity_frame_effect(
			# 	frame_effects, self.ball_velocity_average)

			# CatchingDrillContext.add_ball_2d_critical_points_frame_effect(frame_effects, self.critical_points)

			self.frame_effectss.append(frame_effects)

	def generate_point_projectors(self, cam_intrinsics):
		self.point_projectors = [
			PointProjector.initialize_from_katchet_face_pts(
				cam_intrinsics, katchet_face_pts)
			if katchet_face_pts is not None else None
			for katchet_face_pts in self.katchet_faces
		]

	# Additional data generation functions below -------------------------------
	def generate_hand_2d_positions(self, video_dims):
		assert len(self.pose_landmarkss) > 0

		for pose_landmarks in self.pose_landmarkss:

			if pose_landmarks is None:
				self.left_heel_2d_positions.append(None)
				self.right_heel_2d_positions.append(None)
				continue

			left_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
			right_hand = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]

			vid_w, vid_h = video_dims
			sx_left_hand = left_hand.x * vid_w
			sy_left_hand = left_hand.y * vid_h
			sx_right_hand = right_hand.x * vid_w
			sy_right_hand = right_hand.y * vid_h

			self.left_hand_2d_positions.append(
				np.array([sx_left_hand, sy_left_hand]))
			self.right_hand_2d_positions.append(
				np.array([sx_right_hand, sy_right_hand]))

	def generate_results(self) -> CatchingDrillResults:
		return CatchingDrillResults(
			speed=self.ball_speed_average,
			max_height=self.ball_displacement_max_height,
			angle=0.0,
		)

	@staticmethod
	def filter_ball_2d_positions(ball_2d_positions):
		# TODO: Fill out proper logic here

		# 0 to 15 is good for video 101
		first_position_seen = 0
		last_position_seen = 15

		positions_seen = 0
		filtered_ball_2d_positions = []

		first_ball_2d_seen = None
		last_ball_2d_seen = None

		for ball_2d_position in ball_2d_positions:

			if ball_2d_position is None:
				filtered_ball_2d_positions.append(None)
				continue

			if positions_seen >= first_position_seen and positions_seen <= last_position_seen:
				if positions_seen == first_position_seen:
					first_ball_2d_seen = ball_2d_position
				elif positions_seen == last_position_seen:
					last_ball_2d_seen = ball_2d_position

				filtered_ball_2d_positions.append(ball_2d_position)

			positions_seen += 1

		return filtered_ball_2d_positions, first_ball_2d_seen, last_ball_2d_seen

	@staticmethod
	def generate_heel_2d_positions(video_dims, pose_landmarkss):
		assert len(pose_landmarkss) > 0

		left_heel_2d_positions = []
		right_heel_2d_positions = []
		for pose_landmarks in pose_landmarkss:

			if pose_landmarks is None:
				left_heel_2d_positions.append(None)
				right_heel_2d_positions.append(None)
				continue

			left_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
			right_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

			vid_w, vid_h = video_dims
			sx_left_foot = left_heel.x * vid_w
			sy_left_foot = left_heel.y * vid_h
			sx_right_foot = right_heel.x * vid_w
			sy_right_foot = right_heel.y * vid_h

			left_heel_2d_positions.append(
				np.array([sx_left_foot, sy_left_foot]))
			right_heel_2d_positions.append(
				np.array([sx_right_foot, sy_right_foot]))

		return left_heel_2d_positions, right_heel_2d_positions

	@staticmethod
	def generate_heel_3d_positions(left_heel_2d_positions, right_heel_2d_positions, point_projectors):

		left_heel_3d_positions = []
		right_heel_3d_positions = []
		for left_heel_2d_position, right_heel_2d_position, point_projector in \
				zip(left_heel_2d_positions, right_heel_2d_positions, point_projectors):

			if point_projector is None:
				left_heel_3d_positions.append(None)
				right_heel_3d_positions.append(None)
				continue

			if left_heel_2d_position is not None:
				left_heel_3d_positions.append(
					point_projector.project_2d_to_3d(left_heel_2d_position, Z=0))
			else:
				left_heel_3d_positions.append(None)

			if right_heel_2d_position is not None:
				right_heel_3d_positions.append(
					point_projector.project_2d_to_3d(right_heel_2d_position, Z=0))
			else:
				right_heel_3d_positions.append(None)

		return left_heel_3d_positions, right_heel_3d_positions

	@staticmethod
	def generate_trajectory_plane():
		# TODO: Process ball positions, pick the 3d hand position where the ball
		# is intersecting with the hand to construct the trajectory plane

		# Good for video 849
		# return Plane(
		# 	np.array([0, 0.25, 0]),
		# 	np.array([12, -2, 0]),
		# 	np.array([0, 0.25, -1])
		# )

		# Good for 902
		# return Plane(
		# 	np.array([0, 0.25, 0]),
		# 	np.array([13, 4.5, 0]),
		# 	np.array([0, 0.25, -1])
		# )

		# Good for 00002.mp4
		return Plane(
			np.array([0.2, 0.3, 0]),
			np.array([10, 0.8, 0]),
			np.array([0.2, 0.3, -1])
		)

	@staticmethod
	def generate_angle_between_planes(trajectory_plane, x_plane):
		return np.pi - trajectory_plane.calculate_angle_with_plane(x_plane)

	@staticmethod
	def generate_intersection_point_of_planes(x_plane, trajectory_plane):
		return trajectory_plane.calculate_intersection_point_between_planes(x_plane).reshape((3, 1))

	@staticmethod
	def generate_x_plane():
		return Plane(
			np.array([0, 0, 0]),
			np.array([1, 0, 0]),
			np.array([1, 0, 1])
		)

	@staticmethod
	def generate_ground_plane():
		return Plane(
			np.array([0, 0, 0]),
			np.array([1, 0, 0]),
			np.array([0, 1, 0])
		)

	@staticmethod
	def generate_circle_points():
		circle_initial_point = np.array([2, 0, 0])
		point_rotation_matrix = \
			Plane.get_rotation_matrix_about_point(
				np.pi / 4, np.array([0, 0, 0]), axis="Z")

		circle_points = []
		current_point = np.append(circle_initial_point, 1)
		for x in range(8):
			circle_points.append(np.delete(current_point, -1))
			current_point = point_rotation_matrix @ current_point

		return circle_points

	@staticmethod
	def generate_ball_3d_positions(ball_2d_positions, trajectory_plane, point_projectors):
		ball_3d_positions = []
		for point_projector, ball_2d_position in zip(point_projectors, ball_2d_positions):
			if ball_2d_position is None or point_projector is None:
				ball_3d_positions.append(None)
				continue

			ball_3d_position = point_projector.project_2d_to_3d_plane(
				ball_2d_position, trajectory_plane)

			ball_3d_positions.append(ball_3d_position)

		return ball_3d_positions

	@staticmethod
	def decompose_ball_3d_positions(ball_3d_positions, trajectory_plane):
		ball_decomposed_positions = []
		for ball_3d_position in ball_3d_positions:
			if ball_3d_position is None:
				ball_decomposed_positions.append(None)
				continue

			v1_coeff, v2_coeff = \
				trajectory_plane.decompose_intersecting_point(
					ball_3d_position.reshape((3, )))

			ball_decomposed_positions.append(np.array([v1_coeff, v2_coeff]))

		return ball_decomposed_positions

	def generate_ball_critical_points(self):
		ball_positions = self.ball_detector.get_ball_positions()
		detector = CriticalBallPointDetector(
			ball_positions, self.left_hand_2d_positions, self.right_hand_2d_positions)

		self.critical_points = list(detector.get_critical_points())

		bounce_pt = None
		catch_pt = None

		for pt in self.critical_points:
			match pt.get_type():
				case CriticalPointType.BOUNCE:
					bounce_pt = pt
				case CriticalPointType.CATCH:
					catch_pt = pt

		return bounce_pt, catch_pt

	# Frame effect augmentation below -----------------------------------------------

	@staticmethod
	def add_katchet_face_frame_effect(frame_effects, katchet_face):
		if katchet_face is not None:
			frame_effects.append(FrameEffect(
				primary_label="Katchet Face Poly",
				frame_effect_type=FrameEffectType.KATCHET_FACE_POLY,
				katchet_face_poly=katchet_face,
				colour=(0, 0, 0),
			))

	@staticmethod
	def add_left_heel_3d_frame_effect(frame_effects, left_heel_3d):
		if left_heel_3d is not None:
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINT_3D_SINGLE,
				point_3d_single=left_heel_3d,
				primary_label="Left Heel",
				display_label=FrameEffect.generate_point_string(left_heel_3d),
				show_label=True,
				colour=(255, 0, 0)
			))

	@staticmethod
	def add_right_heel_3d_frame_effect(frame_effects, right_heel_3d):
		if right_heel_3d is not None:
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINT_3D_SINGLE,
				point_3d_single=right_heel_3d,
				primary_label="Right Heel",
				display_label=FrameEffect.generate_point_string(right_heel_3d),
				show_label=True,
				colour=(255, 0, 0)
			))

	@staticmethod
	def add_intersection_point_frame_effect(frame_effects, intersection_point):
		if intersection_point is not None:
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINT_3D_SINGLE,
				point_3d_single=intersection_point,
				primary_label="Intersection point",
				display_label=FrameEffect.generate_point_string(
					intersection_point),
				show_label=True,
				colour=(0, 255, 0)
			))

	@staticmethod
	def add_trajectory_plane_frame_effect(frame_effects, trajectory_plane):
		print("FOO")

		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		if trajectory_plane is not None:
			trajectory_plane_points = trajectory_plane.sample_grid_points(
				20, 1)
			print(trajectory_plane_points)
			trajectory_plane_points = list(
				filter(plane_points_to_print, trajectory_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Trajectory plane points",
				points_3d_multiple=trajectory_plane_points,
				colour=(0, 255, 255),
				show_label=False,
				point_size=4
			))

	@staticmethod
	def add_x_plane_frame_effect(frame_effects, x_plane):
		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		if x_plane is not None:
			x_plane_points = x_plane.sample_grid_points(20, 1)
			x_plane_points = list(
				filter(plane_points_to_print, x_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="X plane points",
				points_3d_multiple=x_plane_points,
				colour=(0, 255, 255),
				show_label=False,
				point_size=4
			))

	@staticmethod
	def add_trajectory_plane_frame_effect(frame_effects, true_trajectory_plane):
		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -3.5 and x >= -2

		if true_trajectory_plane is not None:
			true_trajectory_plane_points = true_trajectory_plane.sample_grid_points(
				35, 0.5)
			true_trajectory_plane_points = list(
				filter(plane_points_to_print, true_trajectory_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="True trajectory plane points",
				points_3d_multiple=true_trajectory_plane_points,
				colour=(0, 255, 255),
				show_label=False,
				point_size=4
			))

	@staticmethod
	def add_ground_plane_frame_effect(frame_effects, ground_plane):
		if ground_plane is not None:
			ground_plane_points = ground_plane.sample_grid_points(10, 1)
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Ground plane points",
				points_3d_multiple=ground_plane_points,
				colour=(255, 0, 0),
				show_label=False,
				point_size=2
			))

	@staticmethod
	def add_angle_printing_frame_effect(frame_effects, angle_rad):
		if angle_rad is not None:
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.TEXT,
				primary_label="Angle between planes",
				display_label=f"t = {(angle_rad / (2 * np.pi)) * 360}",
				show_label=True
			))

			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.TEXT,
				primary_label="Angle between planes",
				display_label=f"180 - t = {180 - ((angle_rad / (2 * np.pi)) * 360)}",
				show_label=True
			))

	@staticmethod
	def add_circle_frame_effect(frame_effects, circle_points):
		if circle_points is not None:
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Circle points",
				points_3d_multiple=circle_points,
				colour=(0, 255, 0),
				show_label=False
			))

	@staticmethod
	def add_katchet_board_points_frame_effect(frame_effects):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
			primary_label="Katchet board points",
			points_3d_multiple=np.array(
				[KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R]),
			colour=(0, 0, 255),
			show_label=True
		))

	# TODO: Add a frame_effect for this
	@staticmethod
	def add_pose_landmarks_frame_effect(frame, pose_landmarks):
		if pose_landmarks is not None:
			mp_drawing.draw_landmarks(
				frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

	@staticmethod
	def add_ball_2d_positions_frame_effect(frame_effects, ball_2d_position, ball_2d_positions_so_far):
		if ball_2d_position is not None:
			ball_2d_positions_so_far.append(ball_2d_position)

		measured_2d_ball_positions = [(x, y) for (
			x, y, isinterpol) in ball_2d_positions_so_far if not isinterpol]
		interpol_2d_ball_positions = [(x, y) for (
			x, y, isinterpol) in ball_2d_positions_so_far if isinterpol]

		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
			primary_label="Ball 2D positions (measured)",
			points_2d_multiple=copy(measured_2d_ball_positions),
			colour=(0, 255, 0),
			show_label=False
		))

		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
			primary_label="Ball 2D positions (interpol)",
			points_2d_multiple=copy(interpol_2d_ball_positions),
			colour=(255, 0, 0),
			show_label=False
		))

	@staticmethod
	def add_ball_3d_positions_frame_effect(frame_effects, ball_3d_position, ball_3d_positions_so_far):
		if ball_3d_position is not None:
			ball_3d_positions_so_far.append(ball_3d_position)

		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
			primary_label="Ball 3D positions",
			points_3d_multiple=copy(ball_3d_positions_so_far),
			colour=(0, 0, 255),
			show_label=True
		))

	@staticmethod
	def add_trajectory_change_2d_positions_frame_effect(frame_effects, first_trajectory_change_2d_position, last_trajectory_change_2d_position):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
			primary_label="Change in trajectory points",
			points_2d_multiple=[
				first_trajectory_change_2d_position, last_trajectory_change_2d_position],
			colour=(0, 255, 0),
			show_label=False
		))

	@staticmethod
	def add_ball_2d_critical_points_frame_effect(frame_effects, critical_points):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
			primary_label="Ball positions",
			points_2d_multiple=copy([cp.get_position_2d()
									for cp in critical_points]),
			colour=(0, 255, 255),
			show_label=False
		))

	@staticmethod
	def add_frame_number(frame_effects, frame_num):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.TEXT,
			primary_label="Frame number",
			display_label=f"F_NUM: {frame_num}",
			show_label=True
		))

	@staticmethod
	def add_ball_average_speed_frame_effect(frame_effects, ball_speed):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.TEXT,
			primary_label="Average speed",
			display_label=f"Average speed: {ball_speed} ms-1",
			show_label=True
		))

	@staticmethod
	def ball_displacement_max_height_frame_effect(frame_effects, max_ball_height):
		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.TEXT,
			primary_label="Max ball height",
			display_label=f"Max ball height: {max_ball_height} m",
			show_label=True
		))
