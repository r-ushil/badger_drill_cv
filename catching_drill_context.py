import mediapipe as mp
import numpy as np

from copy import copy

from frame_effect import FrameEffect, FrameEffectType
from katchet_board import KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from plane import Plane

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingDrillContext():
	ball_positions: list

	def __init__(self) -> None:
		# Exist per frame
		self.frames = []
		self.katchet_faces = []
		self.point_projectors = []
		self.pose_landmarkss = []
		self.ball_positions = []

		self.left_heel_2d_positions = []
		self.right_heel_2d_positions = []

		self.left_heel_3d_positions = []
		self.right_heel_3d_positions = []

		self.trajectory_planes = []
		self.angle_between_planes = []

		# Remain the same throughout the drill
		self.x_plane_fixed = None
		self.ground_plane_fixed = None
		self.circle_points_fixed = None

		self.frame_effectss = []
	
	def generate_augmented_data(self, video_dims):
		self.generate_heel_2d_positions(video_dims)
		self.generate_heel_3d_positions()
		self.generate_trajectory_plane()
		self.generate_x_plane()
		self.generate_ground_plane()
		self.generate_angle_between_planes()
		self.generate_circle_points()
	
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
		for left_heel_3d_position in self.left_heel_3d_positions:

			trajectory_plane = None
			if left_heel_3d_position is not None:
				trajectory_plane = Plane(
					np.array([0, 0.5, 0]),
					np.array(left_heel_3d_position.reshape(3,)),
					np.array([0, 0.5, -1])
				)

			self.trajectory_planes.append(trajectory_plane)

	
	def generate_angle_between_planes(self):
		assert self.x_plane is not None

		for trajectory_plane in self.trajectory_planes:
			angle_between_planes = None
			if trajectory_plane is not None:
				angle_between_planes = trajectory_plane.calculate_angle_with_plane(self.x_plane)
			
			self.angle_between_planes.append(angle_between_planes)

	def generate_x_plane(self):
		self.x_plane = Plane(
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
		
	
	def generate_frame_effects(self):
		ball_positions_so_far = []

		for (frame,
			katchet_face,
			left_heel_3d,
			right_heel_3d,
			trajectory_plane,
			angle_between_planes,
			pose_landmarks,
			ball_position) in zip(
				self.frames, 
				self.katchet_faces, 
				self.left_heel_3d_positions, 
				self.right_heel_3d_positions,
				self.trajectory_planes,
				self.angle_between_planes,
				self.pose_landmarkss,
				self.ball_positions
			):

			frame_effects = []

			if katchet_face is not None:
				frame_effects.append(FrameEffect(
					primary_label="Katchet Face Poly",
					frame_effect_type=FrameEffectType.KATCHET_FACE_POLY,
					katchet_face_poly=katchet_face,
					colour=(0, 0, 0),
				))
			
			self.frame_effectss.append(frame_effects)

			if left_heel_3d is not None:
				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.POINT_3D_SINGLE,
					point_3d_single=left_heel_3d,
					primary_label="Left Heel",
					display_label=FrameEffect.generate_point_string(left_heel_3d),
					show_label=True,
					colour=(255, 0, 0)
				))

			if right_heel_3d is not None:
				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.POINT_3D_SINGLE,
					point_3d_single=right_heel_3d,
					primary_label="Right Heel",
					display_label=FrameEffect.generate_point_string(right_heel_3d),
					show_label=True,
					colour=(255, 0, 0)
				))

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
			
			if self.x_plane is not None:
				x_plane_points = self.x_plane.sample_grid_points(20, 1)
				x_plane_points = list(filter(plane_points_to_print, x_plane_points))
				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
					primary_label="X plane points",
					points_3d_multiple=x_plane_points,
					colour=(0, 0, 255),
					show_label=False
				))
			
			if self.ground_plane_fixed is not None:
				ground_plane_points = self.ground_plane_fixed.sample_grid_points(10, 1)
				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
					primary_label="Ground plane points",
					points_3d_multiple=ground_plane_points,
					colour=(255, 0, 0),
					show_label=False,
					point_size=2
				))

			if angle_between_planes is not None:
				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.TEXT,
					primary_label="Angle between planes",
					display_label=f"t = {(angle_between_planes / (2 * np.pi)) * 360}",
					show_label=True
				))

				frame_effects.append(FrameEffect(
					frame_effect_type=FrameEffectType.TEXT,
					primary_label="Angle between planes",
					display_label=f"180 - t = {180 - ((angle_between_planes / (2 * np.pi)) * 360)}",
					show_label=True
				))
			
			if self.circle_points_fixed is not None:
				frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Circle points",
				points_3d_multiple=self.circle_points_fixed,
				colour=(0, 255, 0),
				show_label=False
			))

			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_3D_MULTIPLE,
				primary_label="Katchet board points",
				points_3d_multiple=np.array([KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R]),
				colour=(0, 0, 255),
				show_label=True
			))

			if pose_landmarks is not None:
				mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
			
			if ball_position is not None:
				ball_positions_so_far.append(ball_position)
			
			frame_effects.append(FrameEffect(
				frame_effect_type=FrameEffectType.POINTS_2D_MULTIPLE,
				primary_label="Ball positions",
				points_2d_multiple=copy(ball_positions_so_far),
				colour=(0, 255, 0),
				show_label=False
			))	
