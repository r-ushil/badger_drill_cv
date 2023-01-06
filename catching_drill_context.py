import mediapipe as mp
import numpy as np

from copy import copy

from frame_effect import FrameEffect, FrameEffectType
from katchet_board import KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from plane import Plane

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingDrillContext():
	ball_2d_positions: list

	def __init__(self) -> None:
		# Exist per frame
		self.frames = []
		self.katchet_faces = []
		self.point_projectors = []
		self.pose_landmarkss = []
		self.ball_2d_positions = []

		self.left_heel_2d_positions = []
		self.right_heel_2d_positions = []

		self.left_heel_3d_positions = []
		self.right_heel_3d_positions = []

		self.ball_3d_positions = []

		# Remain the same throughout the drill
		self.x_plane_fixed = None
		self.ground_plane_fixed = None
		self.circle_points_fixed = None
		self.trajectory_plane_fixed = None
		self.angle_between_planes_fixed = None
		self.intersection_point_of_planes_fixed = None

		self.frame_effectss = []
	
	def generate_augmented_data(self, video_dims):
		self.generate_heel_2d_positions(video_dims)
		self.generate_heel_3d_positions()
		self.generate_trajectory_plane()
		self.generate_x_plane()
		self.generate_ground_plane()
		self.generate_angle_between_planes()
		self.generate_intersection_point_of_planes()
		self.generate_ball_3d_positions()
		# self.generate_circle_points()

	def generate_frame_effects(self):
		ball_2d_positions_so_far = []
		ball_3d_positions_so_far = []

		for (frame,
			katchet_face,
			left_heel_3d,
			right_heel_3d,
			pose_landmarks,
			ball_2d_position,
			ball_3d_position) in zip(
				self.frames, 
				self.katchet_faces, 
				self.left_heel_3d_positions, 
				self.right_heel_3d_positions,
				self.pose_landmarkss,
				self.ball_2d_positions,
				self.ball_3d_positions
			):

			frame_effects = []

			# TODO: Add frame effect for pose landmarks rather than directly
			# annotating the frame
			CatchingDrillContext.add_pose_landmarks_frame_effect(frame, pose_landmarks)
			# CatchingDrillContext.add_ball_2d_positions_frame_effect(frame_effects, ball_2d_position, ball_2d_positions_so_far)
			CatchingDrillContext.add_ball_3d_positions_frame_effect(frame_effects, ball_3d_position, ball_3d_positions_so_far)
			CatchingDrillContext.add_katchet_face_frame_effect(frame_effects, katchet_face)
			CatchingDrillContext.add_katchet_board_points_frame_effect(frame_effects)

			CatchingDrillContext.add_left_heel_3d_frame_effect(frame_effects, left_heel_3d)
			# CatchingDrillContext.add_right_heel_3d_frame_effect(frame_effects, right_heel_3d)

			CatchingDrillContext.add_trajectory_plane_frame_effect(frame_effects, self.trajectory_plane_fixed)
			CatchingDrillContext.add_x_plane_frame_effect(frame_effects, self.x_plane_fixed)
			CatchingDrillContext.add_ground_plane_frame_effect(frame_effects, self.ground_plane_fixed)

			CatchingDrillContext.add_angle_printing_frame_effect(frame_effects, self.angle_between_planes_fixed)
			CatchingDrillContext.add_intersection_point_frame_effect(frame_effects, self.intersection_point_of_planes_fixed)
			# CatchingDrillContext.add_circle_frame_effect(frame_effects, self.circle_points_fixed)

			self.frame_effectss.append(frame_effects)
	
	# Additional data generation functions below -------------------------------
	
	def generate_heel_2d_positions(self, video_dims):
		assert len(self.pose_landmarkss) > 0

		for pose_landmarks in self.pose_landmarkss:

			if pose_landmarks is None:
				self.left_heel_2d_positions.append(None)
				self.right_heel_2d_positions.append(None)
				continue

			left_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
			right_heel = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

			vid_w, vid_h = video_dims
			sx_left_foot = left_heel.x * vid_w
			sy_left_foot = left_heel.y * vid_h
			sx_right_foot = right_heel.x * vid_w
			sy_right_foot = right_heel.y * vid_h

			self.left_heel_2d_positions.append(np.array([sx_left_foot, sy_left_foot]))
			self.right_heel_2d_positions.append(np.array([sx_right_foot, sy_right_foot]))
	
	def generate_heel_3d_positions(self):
		
		for left_heel_2d_position, right_heel_2d_position, point_projector in \
			zip(self.left_heel_2d_positions, self.right_heel_2d_positions, self.point_projectors):

			if point_projector is None:
				self.left_heel_3d_positions.append(None)
				self.right_heel_3d_positions.append(None)
				continue
			
			if left_heel_2d_position is not None:
				self.left_heel_3d_positions.append(point_projector.project_2d_to_3d(left_heel_2d_position, Z=0))
			else:
				self.left_heel_3d_positions.append(None)

			if right_heel_2d_position is not None:
				self.right_heel_3d_positions.append(point_projector.project_2d_to_3d(right_heel_2d_position, Z=0))
			else:
				self.right_heel_3d_positions.append(None)
	
	def generate_trajectory_plane(self):
		# TODO: Process ball positions, pick the 3d hand position where the ball 
		# is intersecting with the hand to construct the trajectory plane

		# Good for video 849
		# self.trajectory_plane_fixed = Plane(
		# 	np.array([0, 0.25, 0]),
		# 	np.array([12, -2, 0]),
		# 	np.array([0, 0.25, -1])
		# )

		# Good for 902
		# self.trajectory_plane_fixed = Plane(
		# 	np.array([0, 0.25, 0]),
		# 	np.array([13, 4.5, 0]),
		# 	np.array([0, 0.25, -1])
		# )

		# Good for 101
		self.trajectory_plane_fixed = Plane(
			np.array([0, 0.25, 0]),
			np.array([10, -0.15, 0]),
			np.array([0, 0.25, -1])
		)
	
	def generate_angle_between_planes(self):
		assert self.x_plane_fixed is not None
		assert self.trajectory_plane_fixed is not None

		self.angle_between_planes_fixed = \
			np.pi - self.trajectory_plane_fixed.calculate_angle_with_plane(self.x_plane_fixed)

	def generate_intersection_point_of_planes(self):
		assert self.x_plane_fixed is not None
		assert self.trajectory_plane_fixed is not None

		self.intersection_point_of_planes_fixed = \
			self.trajectory_plane_fixed.calculate_intersection_point_between_planes(self.x_plane_fixed).reshape((3, 1))

	def generate_x_plane(self):
		self.x_plane_fixed = Plane(
			np.array([0, 0, 0]),
			np.array([1, 0, 0]),
			np.array([1, 0, 1])
		)

	def generate_ground_plane(self):
		self.ground_plane_fixed = Plane(
			np.array([0, 0, 0]),
			np.array([1, 0, 0]),
			np.array([0, 1, 0])
		)
	
	def generate_circle_points(self):
		circle_initial_point = np.array([2, 0, 0])
		point_rotation_matrix = \
			Plane.get_rotation_matrix_about_point(np.pi / 4, np.array([0, 0, 0]), axis="Z")

		circle_points = []
		current_point = np.append(circle_initial_point, 1)
		for x in range(8):
			circle_points.append(np.delete(current_point, -1))
			current_point = point_rotation_matrix @ current_point 
		
		self.circle_points_fixed = circle_points
	
	def generate_ball_3d_positions(self):
		assert self.angle_between_planes_fixed is not None
		assert self.intersection_point_of_planes_fixed is not None

		for point_projector, ball_2d_position in zip(self.point_projectors, self.ball_2d_positions):
			if ball_2d_position is None or point_projector is None:
				self.ball_3d_positions.append(None)
				continue
			
			# TODO: Ensure these rotation matrices are around the correct axis
			rot_mat = Plane.get_rotation_matrix_about_point(
				self.angle_between_planes_fixed,
				self.intersection_point_of_planes_fixed.reshape((3, ))
			)

			rot_mat_inv = Plane.get_rotation_matrix_about_point(
				-self.angle_between_planes_fixed,
				self.intersection_point_of_planes_fixed.reshape((3, ))
			)
			
			rotated_point_projector = point_projector.create_rotated_camera(rot_mat)

			# TODO: Ensure that this axis is correct
			ball_3d_position_rotated = rotated_point_projector.project_2d_to_3d(ball_2d_position, X=0)	

			# TODO: Ensure that this rotation matrix is multiplied in the correct order
			ball_3d_position = Plane.multiply_orthogonal_matrix_by_non_orthogonal_vec(rot_mat_inv, ball_3d_position_rotated)

			self.ball_3d_positions.append(ball_3d_position)
	
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
				display_label=FrameEffect.generate_point_string(intersection_point),
				show_label=True,
				colour=(0, 255, 0)
			))

	@staticmethod
	def add_trajectory_plane_frame_effect(frame_effects, trajectory_plane):
		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		if trajectory_plane is not None:
			trajectory_plane_points = trajectory_plane.sample_grid_points(20, 1)
			trajectory_plane_points = list(filter(plane_points_to_print, trajectory_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Trajectory plane points",
				points_3d_multiple=trajectory_plane_points,
				colour=(0, 0, 255),
				show_label=False
			))
	
	@staticmethod
	def add_x_plane_frame_effect(frame_effects, x_plane):
		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		if x_plane is not None:
			x_plane_points = x_plane.sample_grid_points(20, 1)
			x_plane_points = list(filter(plane_points_to_print, x_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="X plane points",
				points_3d_multiple=x_plane_points,
				colour=(0, 255, 0),
				show_label=False
			))

	@staticmethod
	def add_trajectory_plane_frame_effect(frame_effects, true_trajectory_plane):
		def plane_points_to_print(point):
			(x, _, z) = point
			return z <= 0 and z > -0.5 and x >= -2

		if true_trajectory_plane is not None:
			true_trajectory_plane_points = true_trajectory_plane.sample_grid_points(20, 1)
			true_trajectory_plane_points = list(filter(plane_points_to_print, true_trajectory_plane_points))
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="True trajectory plane points",
				points_3d_multiple=true_trajectory_plane_points,
				colour=(0, 255, 0),
				show_label=False
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
			points_3d_multiple=np.array([KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R]),
			colour=(0, 0, 255),
			show_label=True
		))

	# TODO: Add a frame_effect for this
	@staticmethod
	def add_pose_landmarks_frame_effect(frame, pose_landmarks):
		if pose_landmarks is not None:
			mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)

	@staticmethod
	def add_ball_2d_positions_frame_effect(frame_effects, ball_2d_position, ball_2d_positions_so_far):
		if ball_2d_position is not None:
			ball_2d_positions_so_far.append(ball_2d_position)

		frame_effects.append(FrameEffect(
			frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
			primary_label="Ball 2D positions",
			points_2d_multiple=copy(ball_2d_positions_so_far),
			colour=(0, 255, 0),
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
			show_label=False
		))