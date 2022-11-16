import mediapipe as mp
import numpy as np
import cv2
from enum import Enum

SHOULDER_WIDTH_THRESHOLD = 0.1
HAND_HIP_THRESHOLD = 0.1
VERTICAL_ALIGNMENT_THRESHOLD = 0.05
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class Stance(Enum):
	READY = 0,
	PRE_SHOT = 1,
	POST_SHOT = 2,
	TRANSITION = 3,

class CoverDriveJudge():
	def __init__(self, input_video_path):
		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False, 
			min_detection_confidence=0.5, 
			min_tracking_confidence=0.5, 
			model_complexity=2
		)

		self.video_capture = cv2.VideoCapture(input_video_path)

		if not self.video_capture.isOpened():
			print("Error opening video file")
			raise TypeError

		self.frame_width = int(self.video_capture.get(3))
		self.frame_height = int(self.video_capture.get(4))
		fps = int(self.video_capture.get(5))

		# setup output video 
		output_video_path = self.generate_output_video_path(input_video_path)

		self.video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
		'm', 'p', '4', 'v'), fps, (self.frame_width, self.frame_height))
  
	def process_and_write_video(self):
		frame_present, frame = self.video_capture.read()

		# create empty scores array to store scores for each frame
		scores = np.zeros(3)
		frames_processed = 0

		while frame_present:
			(stance, score) = self.process_and_write_frame(frame)

			# add score calculated to scores array
			if score != None:
				scores[stance.value] += score
				frames_processed += 1

			frame_present, frame = self.video_capture.read()

		# calculate average score for all stances
		avgScore = np.sum(scores) / (frames_processed * 3)
  
	def process_and_write_frame(self, image):
		# convert colour format from BGR to RBG
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		image.flags.writeable = False

		# run pose estimation on frame
		landmark_results = self.pose_estimator.process(image)

		# convert colour format back to BGR
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# write pose landmarks from results onto frame
		mp_drawing.draw_landmarks(image, landmark_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# TODO: - add logic to check that these landmarks are actually detected. (i.e. if landmark_results.pose_landmarks is None)

		# check if the player is in the ready stance
		ready_stance = self.is_ready(landmark_results.pose_landmarks.landmark)
		# check if the player is in the pre-shot stance
		pre_shot_stance = self.is_pre_shot(landmark_results.pose_landmarks.landmark)
		# check if the player is in the post-shot stance
		post_shot_stance = self.is_post_shot(landmark_results.pose_landmarks.landmark)

		score_with_stance = self.score_stance(landmark_results.pose_landmarks.landmark)

		image = cv2.flip(image, 0)

		cv2.putText(image, f'Ready Stance: {ready_stance}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(image, f'Pre Shot Stance: {pre_shot_stance}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(image, f'Post Shot Stance: {post_shot_stance}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

		self.video_writer.write(image)
		return score_with_stance

	# scores stance based on landmarks, and returns shot classification and score
	def score_stance(self, landmarks):
		# if the player is in the ready stance, score relative to ready stance
		if self.is_ready(landmarks):
			return (Stance.READY, self.score_ready_stance(landmarks))
		# if the player is in the pre-shot stance, return 2
		elif self.is_pre_shot(landmarks):
			#TODO
			return (Stance.PRE_SHOT, None)
		# if the player is in the post-shot stance, return 3
		elif self.is_post_shot(landmarks):
			#TODO
			return (Stance.POST_SHOT, None)
		# if the player is in none of the stances, the player is transistioning between stances
		else:
			#TODO
			return (Stance.TRANSITION, None)

	def score_ready_stance(self, landmarks):
		shoulder_feet_difference = self.difference_in_feet_and_shoulder_width(
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
			landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
		)

		hand_hip_displacement = self.hand_hip_displacement(
			landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
			landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
		)

		if shoulder_feet_difference is None or hand_hip_displacement is None:
			return None

		# normalise shoulder_feet_difference to 0-1, using SHOULDER_WIDTH_THRESHOLD
		weighting = 1 / SHOULDER_WIDTH_THRESHOLD
		shoulder_feet_score = (SHOULDER_WIDTH_THRESHOLD - shoulder_feet_difference) * weighting

		# normalise hand_hip_displacement to 0-1, using HAND_HIP_THRESHOLD
		weighting = 1 / HAND_HIP_THRESHOLD
		hand_hip_score = (HAND_HIP_THRESHOLD - hand_hip_displacement) * weighting

		return (shoulder_feet_score + hand_hip_score) / 2

	# returns true if any landmarks of interest for a given frame have low visibility
	@staticmethod
	def ignore_low_visibility(landmarks):
		return any(landmark.visibility < 0.7 for landmark in landmarks)

	# calculates the x displacement between two landmarks
	@staticmethod
	def calculate_x_displacement(a, b):
		return abs(a.x - b.x)

	# checks whether landmarks are vertically aligned, within a threshold
	@staticmethod
	def is_vertically_aligned(top, middle, bottom, threshold):
		x1 = CoverDriveJudge.calculate_x_displacement(top	, middle)
		x2 = CoverDriveJudge.calculate_x_displacement(middle, bottom)
		return (x1 < threshold) and (x2 < threshold)


	# checks whether the player is in the post-shot stance
	def is_post_shot(self, landmarks):
		return CoverDriveJudge.is_vertically_aligned(
			landmarks[mp_pose.PoseLandmark.NOSE],
			landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
			landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
			VERTICAL_ALIGNMENT_THRESHOLD,
		)

	# checks whether the player is in the pre-shot stance
	# TODO: check elbow angle with shoulder
	def is_pre_shot(self, landmarks):
		return CoverDriveJudge.is_vertically_aligned(
			landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
			landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
			landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
			VERTICAL_ALIGNMENT_THRESHOLD,
		)

	# checks whether the player is in the ready stance
	def is_ready(self, landmarks):

		# true if feet are shoulder width apart, within a threshold
		feet_shoulder_width_apart = self.feet_shoulder_width_apart(
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
			landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
		)

		# true if hands are close to hips along the x axis, within a threshold
		hand_close_to_hips = self.hand_close_to_hips(
			landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
			landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
		)

		return feet_shoulder_width_apart and hand_close_to_hips


	# calculates difference between feet width and shoulder width, returning None if visibility is too low
	def difference_in_feet_and_shoulder_width(self, left_shoulder, right_shoulder, left_ankle, right_ankle):
		if CoverDriveJudge.ignore_low_visibility([left_shoulder, right_shoulder, left_ankle, left_ankle]):
			return None

		# calculate the distance between the left and right shoulders
		shoulder_width = self.calculate_x_displacement(left_shoulder, right_shoulder)
		# calculate the distance between the left and right ankles
		feet_width = self.calculate_x_displacement(left_ankle, right_ankle)
		# return the difference between the shoulder and feet width
		return abs(shoulder_width - feet_width)

	# Returns a boolean on whether the feet are shoulder width apart
	def feet_shoulder_width_apart(self, left_shoulder, right_shoulder, left_foot, right_foot):
		difference = self.difference_in_feet_and_shoulder_width(left_shoulder, right_shoulder, left_foot, right_foot)
		return (difference is not None and difference < SHOULDER_WIDTH_THRESHOLD)

	def hand_hip_displacement(self, hip, hand):
		# assume the hands aren't on the hips if the landmarks aren't detected
		if CoverDriveJudge.ignore_low_visibility([hip, hand]):
			return None

		return CoverDriveJudge.calculate_x_displacement(hip, hand)

	def hand_close_to_hips(self, hip, hand):
		# calculate the x displacement between the hands and the hips
		displacement = self.hand_hip_displacement(hip, hand)
		return (displacement is not None and displacement < HAND_HIP_THRESHOLD)
  
	@staticmethod
	def generate_output_video_path(input_video_path):
		output_video_directory, filename = \
			input_video_path[:input_video_path.rfind('/')+1], input_video_path[input_video_path.rfind('/')+1:]
  
		input_video_filename, input_video_extension = filename.split('.')
  
		output_video_path = f'{output_video_directory}{input_video_filename}_annotated.{input_video_extension}'
  
		return output_video_path

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.pose_estimator.close()
		self.video_capture.release()
		self.video_writer.release()
