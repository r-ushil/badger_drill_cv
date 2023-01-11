import mediapipe as mp
import numpy as np
import cv2
from enum import Enum

SHOULDER_WIDTH_THRESHOLD = 0.15
HAND_HIP_THRESHOLD = 0.15
VERTICAL_ALIGNMENT_THRESHOLD = 0.1
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class Stance(Enum):
	READY = 0
	PRE_SHOT = 1
	POST_SHOT = 2
	TRANSITION = 3

# TODO: Associate metrics with the drill stage from which they originated, remove
# hard coded logic
class Metrics(Enum):
	HAND_BY_HIP = 0
	FEET_SHOULDER_WIDTH = 1
	BACKLIFT = 2
	DROPPED_SHOULDER = 3
	HEAD_KNEE_ALIGNMENT = 4
	ELBOW_ANGLES = 5


def get_advice(metric: Metrics):
	match metric:
		case Metrics.HAND_BY_HIP:
			return "Keep your hands by your hips"
		case Metrics.FEET_SHOULDER_WIDTH:
			return "Keep your feet shoulder width apart"
		case Metrics.BACKLIFT:
			return "Watch your backlift"
		case Metrics.DROPPED_SHOULDER:
			return "Watch your dropped shoulder"
		case Metrics.HEAD_KNEE_ALIGNMENT:
			return "Watch your head-knee alignment"
		case Metrics.ELBOW_ANGLES:
			return "Watch your elbow angles"
			
class CoverDriveJudge():
	def __init__(self, input_video_path, no_output=False):
		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False, 
			min_detection_confidence=0.5, 
			min_tracking_confidence=0.5, 
			model_complexity=2
		)

		self.video_capture = cv2.VideoCapture(input_video_path)
		self.scores = np.zeros(len(Metrics))
		self.frames_processed = np.zeros(len(Metrics))

		if not self.video_capture.isOpened():
			print("Error opening video file")
			raise TypeError

		self.frame_width = int(self.video_capture.get(3))
		self.frame_height = int(self.video_capture.get(4))
		fps = int(self.video_capture.get(5))

		if no_output:
			self.video_writer = None
		else:
			output_video_path = CoverDriveJudge.generate_output_video_path(input_video_path)
			self.video_writer = cv2.VideoWriter(
				output_video_path,
				cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
				fps,
				(self.frame_width, self.frame_height),
			)
  
	def process_and_write_video(self):
		frame_present, frame = self.video_capture.read()

		# create empty scores array to store scores for each frame
		scores = np.zeros(3)
		frames_processed = np.zeros(3)

		while frame_present:
			self.process_and_write_frame(frame)
			frame_present, frame = self.video_capture.read()

		return self.display_scores_and_advice()

	def display_scores_and_advice(self):

		# In the case that 0 frames are processed for a score, the score
		# will be zero, as will frame_processed, resulting in a score of 0/0=nan
		# convert this to a score of 0

		# convert 0s to 1s in frames_processed to avoid 0 / 0 (divide by zero error)
		self.frames_processed[self.frames_processed == 0] = 1

		averageScores = np.nan_to_num(self.scores / self.frames_processed)
		
		stance_scores = np.zeros(3)
		stance_scores[Stance.READY.value] = (averageScores[Metrics.HAND_BY_HIP.value] + averageScores[Metrics.FEET_SHOULDER_WIDTH.value]) / 2
		stance_scores[Stance.PRE_SHOT.value] = (averageScores[Metrics.BACKLIFT.value] + averageScores[Metrics.DROPPED_SHOULDER.value]) / 2
		stance_scores[Stance.POST_SHOT.value] = (averageScores[Metrics.HEAD_KNEE_ALIGNMENT.value] + averageScores[Metrics.ELBOW_ANGLES.value]) / 2

		# print out the average scores for each stance
		print("\nAverage scores for each stance:")
		print("Ready stance: " + str(stance_scores[Stance.READY.value]))
		print("Pre-shot stance: " + str(stance_scores[Stance.PRE_SHOT.value]))
		print("Post-shot stance: " + str(stance_scores[Stance.POST_SHOT.value]))

		removed_zeros = stance_scores[np.nonzero(stance_scores)]

		print("\nAverage score:")
		average = np.sum(removed_zeros) / len(removed_zeros)
		print(average)

		# get minimum two elements from self.scores, as a two element array in the form of indices
		worst_two_score_indices = np.argpartition(averageScores, 2)[:2]
		
		# print out the advice for the player
		print("\nAdvice for player:")

		worst_advice = get_advice(Metrics(worst_two_score_indices[0]))
		penultimate_worst_advice = (get_advice(Metrics(worst_two_score_indices[1])))
		print(worst_advice)
		print(penultimate_worst_advice)

		return (average, worst_advice, penultimate_worst_advice)
  
	def process_and_write_frame(self, frame):
		# convert colour format from BGR to RBG
		frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
		frame.flags.writeable = False

		# run pose estimation on frame
		landmark_results = self.pose_estimator.process(frame)

		# convert colour format back to BGR
		frame.flags.writeable = True
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		# write pose landmarks from results onto frame
		mp_drawing.draw_landmarks(frame, landmark_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# TODO: - add logic to check that these landmarks are actually detected. (i.e. if landmark_results.pose_landmarks is None)

		self.score_stance(landmark_results.pose_landmarks.landmark)

		frame = cv2.flip(frame, 0)

		if self.video_writer != None:
			self.video_writer.write(frame)

	# scores stance based on landmarks, and returns shot classification and score
	def score_stance(self, landmarks):
		# if the player is in the ready stance, score relative to ready stance
		if self.is_ready(landmarks):
			self.score_ready_stance(landmarks)
		# if the player is in the pre-shot stance, score relative to pre-shot stance
		elif self.is_pre_shot(landmarks) and not self.is_post_shot(landmarks):
			self.score_pre_shot_stance(landmarks)
		# if the player is in the post-shot stance, score relative to post-shot stance
		elif self.is_post_shot(landmarks):
			self.score_post_shot_stance(landmarks)

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

		self.scores[Metrics.FEET_SHOULDER_WIDTH.value] += shoulder_feet_score
		self.frames_processed[Metrics.FEET_SHOULDER_WIDTH.value] += 1
		
		self.scores[Metrics.HAND_BY_HIP.value] += hand_hip_score
		self.frames_processed[Metrics.HAND_BY_HIP.value] += 1

	def score_pre_shot_stance(self, landmarks):	
		IDEAL_BACKLIFT_ANGLE = 180
		IDEAL_BACKLIFT_ANGLE_THRESHOLD = 20
		IDEAL_DISTANCE_BETWEEN_SHOULDERS = 0.04
		DROPPED_SHOULDER_THRESHOLD = 0.03

		backlift_angle = CoverDriveJudge.calculate_angle(
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
		)

		difference_in_shoulder_height = CoverDriveJudge.calculate_y_displacement(
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
		)

		distance_from_ideal_backlift = abs(IDEAL_BACKLIFT_ANGLE - backlift_angle)
		if distance_from_ideal_backlift == 0:
			backlift_score = 1
		elif distance_from_ideal_backlift > IDEAL_BACKLIFT_ANGLE_THRESHOLD:
			backlift_score = 0
		else:
			backlift_score = 1 - (distance_from_ideal_backlift / IDEAL_BACKLIFT_ANGLE_THRESHOLD)

		distance_from_ideal_shoulder = abs(IDEAL_DISTANCE_BETWEEN_SHOULDERS - difference_in_shoulder_height)
		if distance_from_ideal_shoulder == 0:
			shoulder_score = 1
		elif distance_from_ideal_shoulder > DROPPED_SHOULDER_THRESHOLD:
			shoulder_score = 0
		else:
			shoulder_score = 1 - (distance_from_ideal_shoulder / DROPPED_SHOULDER_THRESHOLD)

		self.scores[Metrics.BACKLIFT.value] += backlift_score
		self.frames_processed[Metrics.BACKLIFT.value] += 1

		self.scores[Metrics.DROPPED_SHOULDER.value] += shoulder_score
		self.frames_processed[Metrics.DROPPED_SHOULDER.value] += 1


	def score_post_shot_stance(self, landmarks):
		head_knee_alignment = CoverDriveJudge.calculate_x_displacement(
			landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT],
			landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
		)

		alignment_threshold = 0.1

		if head_knee_alignment > alignment_threshold:
			head_knee_score = 0
		else:
			weighting = 1 / alignment_threshold
			head_knee_score = (alignment_threshold - head_knee_alignment) * weighting
		
		# angle is between 90 and 180 degrees, 90 is ideal
		left_arm_angle = CoverDriveJudge.calculate_angle(
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
			landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
		)

		right_arm_angle = CoverDriveJudge.calculate_angle(
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
			landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
		)

		# normalise angle to 0-1
		weighting = 1 / 90
		left_arm_score = (left_arm_angle - 90) * weighting
		right_arm_score = (right_arm_angle - 90) * weighting
		elbow_angles_score = (left_arm_score + right_arm_score) / 2

		self.scores[Metrics.HEAD_KNEE_ALIGNMENT.value] += head_knee_score
		self.frames_processed[Metrics.HEAD_KNEE_ALIGNMENT.value] += 1

		self.scores[Metrics.ELBOW_ANGLES.value] += elbow_angles_score
		self.frames_processed[Metrics.ELBOW_ANGLES.value] += 1
	

	# returns true if any landmarks of interest for a given frame have low visibility
	@staticmethod
	def ignore_low_visibility(landmarks):
		return any(landmark.visibility < 0.6 for landmark in landmarks)

	# calculates the x displacement between two landmarks
	@staticmethod
	def calculate_x_displacement(a, b):
		return abs(a.x - b.x)

	# calculates the y displacement between two landmarks
	@staticmethod
	def calculate_y_displacement(a, b):
		return abs(a.y - b.y)

	# checks whether landmarks are vertically aligned, within a threshold
	@staticmethod
	def is_vertically_aligned(top, middle, bottom, threshold):
		x1 = CoverDriveJudge.calculate_x_displacement(top	, middle)
		x2 = CoverDriveJudge.calculate_x_displacement(middle, bottom)
		return (x1 < threshold) and (x2 < threshold)

	#calculates angle between three joints
	@staticmethod
	def calculate_angle(a, b, c):
		ab = np.array([a.x - b.x, a.y - b.y])
		cb = np.array([c.x - b.x, c.y - b.y])
		cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
		angle = np.arccos(cosine_angle)
		return np.degrees(angle)

	def is_correct_angle(self, a, b, c, start_threshold, end_threshold):
		# if any of the landmarks are not visible, return false
		if self.ignore_low_visibility([a, b, c]):
			return False
		
		return self.calculate_angle(a, b, c) > start_threshold and self.calculate_angle(a, b, c) < end_threshold

	# checks whether the player is in the post-shot stance
	def is_post_shot(self, landmarks):
		return CoverDriveJudge.is_vertically_aligned(
			landmarks[mp_pose.PoseLandmark.NOSE],
			landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
			landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
			VERTICAL_ALIGNMENT_THRESHOLD,
		)

	# checks whether the player is in the pre-shot stance
	def is_pre_shot(self, landmarks):
		shoulder_angle_with_heel = self.is_correct_angle(
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
			landmarks[mp_pose.PoseLandmark.RIGHT_HEEL],
			100,
			170,
		)

		elbow_angle_with_shoulder = self.is_correct_angle(
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
			80, #TODO: extract into constants, this is what works best
			200,
		)

		return shoulder_angle_with_heel and elbow_angle_with_shoulder

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

		if self.video_writer != None:
			self.video_writer.release()
