import mediapipe as mp
import numpy as np

from frame_effect import FrameEffect, FrameEffectType

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingDrillContext():
	ball_positions: list

	def __init__(self) -> None:
		self.frames = []
		self.katchet_faces = []
		self.point_projectors = []
		self.pose_landmarkss = []
		self.ball_positions = []

		self.left_heel_2d_positions = []
		self.right_heel_2d_positions = []

		self.left_heel_3d_positions = []
		self.right_heel_3d_positions = []

		self.frame_effectss = []
	
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
	
	def generate_frame_effects(self):
		for frame, katchet_face in zip(self.frames, self.katchet_faces):
			frame_effects = []

			if katchet_face is not None:
				frame_effects.append(FrameEffect(
					primary_label="Katchet Face Poly",
					frame_effect_type=FrameEffectType.KATCHET_FACE_POLY,
					katchet_face_poly=katchet_face,
					colour=(0, 0, 0),
				))
			
			self.frame_effectss.append(frame_effects)



