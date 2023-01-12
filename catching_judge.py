from typing import List
import numpy as np
import mediapipe as mp
import cv2

from numpy import array
from catching_drill_results import CatchingDrillError, CatchingDrillResults
from plane import Plane
from point_projector import CameraIntrinsics, PointProjector

from judge import Judge
from katchet_board import KATCHET_BOX_BOT_L, KATCHET_BOX_BOT_R, KATCHET_BOX_TOP_L, KATCHET_BOX_TOP_R
from frame_effect import FrameEffectType, FrameEffect
from catching_drill_context import CatchingDrillContext
from augmented_frame import AugmentedFrame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class CatchingJudge(Judge):
	__cam_pose_estimator: PointProjector

	def __init__(self, input_video_path, cam_intrinsics: CameraIntrinsics, no_output: bool = False):
		super().__init__(input_video_path, no_output)

		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			model_complexity=2,    
		)

		self.__cam_intrinsics = cam_intrinsics

	def process_and_write_video(self):
		drill_context = CatchingDrillContext(self.fps, self.get_video_dims())

		for frame in self.get_frames():
			self.process_frame(drill_context, frame)

		drill_context.interpolate_missing_data()

		katchet_faces = drill_context.katchet_faces
		katchet_faces_len = len(katchet_faces)
		katchet_faces_detected = sum([1 for x in katchet_faces if x is not None])
		proportion_detected = katchet_faces_detected / katchet_faces_len
		if proportion_detected < 0.3:
			return CatchingDrillResults(err=CatchingDrillError.KATCHET_BOARD_NOT_DETECTED)

		drill_context.generate_augmented_data(self.get_video_dims(), cam_intrinsics=self.__cam_intrinsics)
		drill_context.generate_frame_effects()

		for output_frame in self.generate_output_frames(drill_context):
			self.write_frame(output_frame)

		return drill_context.generate_output_results()

	def process_frame(self, drill_context: CatchingDrillContext, frame):
		drill_context.frames.append(frame)

		augmented_frame = AugmentedFrame(frame)

		katchet_face_pts = self.detect_katchet_board(augmented_frame)
		drill_context.katchet_faces.append(katchet_face_pts)

		drill_context.ball_detector.process(augmented_frame)

		drill_context.pose_landmarkss.append(self.detect_pose(augmented_frame))

	def detect_pose(self, augmented_frame: AugmentedFrame):
		return self.pose_estimator.process(augmented_frame.frame_rgb()).pose_landmarks

	def detect_katchet_board(self, augmented_frame: AugmentedFrame):
		# convert colour format from BGR to RBG
		frame = augmented_frame.frame_hsv()
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

		katchet_face_pts = np.reshape(katchet_face_poly, (4, 2))

		return katchet_face_pts

	def generate_output_frames(self, drill_context: CatchingDrillContext) -> cv2.Mat:
		annotated_frames = []
		for frame, point_projector, frame_effects in zip(drill_context.frames, drill_context.point_projectors, drill_context.frame_effectss):

			if point_projector is not None:
				label_counter = 1
				for effect in frame_effects:
					match effect.frame_effect_type:
						case FrameEffectType.POINTS_3D_MULTIPLE:
							for point_3d in effect.points_3d_multiple:
								CatchingJudge.__label_3d_point(point_projector, point_3d, frame, None, effect.show_label, colour=effect.colour, point_size=effect.point_size)
						case FrameEffectType.POINTS_2D_MULTIPLE:
							for point_2d in effect.points_2d_multiple:
								CatchingJudge.__label_2d_point(point_2d, frame, effect.display_label, effect.show_label, colour=effect.colour)
						case FrameEffectType.POINT_3D_SINGLE:
							CatchingJudge.__label_3d_point(point_projector, effect.point_3d_single, frame, effect.display_label, show_label=effect.show_label, colour=effect.colour)
						case FrameEffectType.KATCHET_FACE_POLY:
							cv2.drawContours(
								image=frame,
								contours=[effect.katchet_face_poly],
								contourIdx=0,
								color=effect.colour,
								thickness=8,
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
			else:
				label_counter = 1
				for effect in frame_effects:
					match effect.frame_effect_type:
						case FrameEffectType.POINTS_2D_MULTIPLE:
							for point_2d in effect.points_2d_multiple:
								CatchingJudge.__label_2d_point(point_2d, frame, effect.display_label, effect.show_label, colour=effect.colour)

			annotated_frames.append(frame)

		return annotated_frames

	@staticmethod
	def __label_3d_point(pose_estimator: PointProjector, point_3d: np.ndarray[(3, 1), np.float64], frame, label: str, show_label = True, colour = (255, 0, 0), point_size = 10):
		wx, wy, wz = point_3d[0], point_3d[1], point_3d[2]

		point_2d = pose_estimator.project_3d_to_2d(point_3d).astype('int')

		if show_label and label is None:
			label = FrameEffect.generate_point_string(point_3d)
		
		CatchingJudge.__label_2d_point(point_2d, frame, label, show_label, colour, point_size)
	
	@staticmethod
	def __label_2d_point(point_2d: np.ndarray[(2, 1), np.float64], frame, label, show_label, colour, point_size = 10):
		try:
			sx, sy = point_2d[0], point_2d[1]
			cv2.circle(frame, (sx, sy), point_size, colour, -1)
			if show_label:
				cv2.putText(
					frame,
					label if label is not None else FrameEffect.generate_point_string(point_2d),
					(sx, sy),
					fontFace=cv2.FONT_HERSHEY_SIMPLEX,
					fontScale=0.5,
					color=(255, 255, 255),
					thickness=2,
					lineType=cv2.LINE_AA,
				)
		except:
			print("Failed to print")