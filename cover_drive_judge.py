import mediapipe as mp
import numpy as np
import cv2

SHOULDER_WIDTH_THRESHOLD = 0.05
HAND_HIP_THRESHOLD = 0.1
VERTICAL_ALIGNMENT_THRESHOLD = 0.05
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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
		while frame_present:
			self.process_frame(frame)

			frame_present, frame = self.video_capture.read()
  
	def process_frame(self, image):
		# convert colour format from BGR to RBG
		image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
		image.flags.writeable = False

		# run pose estimation on frame
		# TODO: make name more specific
		results = self.pose_estimator.process(image)

		# convert colour format back to BGR
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		# write pose landmarks from results onto frame
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# TODO: - add logic to check that these landmarks are actually detected.

		# check if the player is in the ready stance
		ready_stance = self.check_ready_stance(results.pose_landmarks.landmark)
		pre_shot_stance = self.check_pre_stance(results.pose_landmarks.landmark)
		post_shot_stance = self.check_post_stance(results.pose_landmarks.landmark)

		image = cv2.flip(image, 0)

		cv2.putText(image, f'Ready Stance: {ready_stance}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(image, f'Pre Shot Stance: {pre_shot_stance}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
		cv2.putText(image, f'Post Shot Stance: {post_shot_stance}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

		self.video_writer.write(image)


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
	def check_vertical_alignment(top, middle, bottom, threshold):
		x1 = CoverDriveJudge.calculate_x_displacement(top	, middle)
		x2 = CoverDriveJudge.calculate_x_displacement(middle, bottom)
		return (x1 < threshold) and (x2 < threshold)

	def check_post_stance(self, landmarks):
		return CoverDriveJudge.check_vertical_alignment(
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE],
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
			VERTICAL_ALIGNMENT_THRESHOLD,
		)

	def check_pre_stance(self, landmarks):
		return CoverDriveJudge.check_vertical_alignment(
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
			landmarks.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
			VERTICAL_ALIGNMENT_THRESHOLD,
		)

	def check_ready_stance(self, landmarks):

		shoulder_width = self.feet_shoulder_width_apart(
			landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
			landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
		)

		hands_on_hips = self.hands_on_hips(
			landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
			landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
		)

		return shoulder_width and hands_on_hips


	# Returns a boolean on whether the feet are shoulder width apart
	def feet_shoulder_width_apart(self, left_shoulder, right_shoulder, left_foot, right_foot):

		# assume the feet aren't shoulder width apart if the landmarks aren't detected
		if CoverDriveJudge.ignore_low_visibility([left_shoulder, right_shoulder, left_foot, right_foot]):
			return False

		# calculate the x displacement between the feet and the shoulders
		shoulder_width = CoverDriveJudge.calculate_x_displacement(left_shoulder, right_shoulder)
		feet_width = CoverDriveJudge.calculate_x_displacement(left_foot, right_foot)

		return abs(shoulder_width - feet_width) < SHOULDER_WIDTH_THRESHOLD

	def hands_on_hips(self, hip, hand):
		# assume the hands aren't on the hips if the landmarks aren't detected
		if CoverDriveJudge.ignore_low_visibility([hip, hand]):
			return False

		# calculate the x displacement between the hands and the hips
		displacement = CoverDriveJudge.calculate_x_displacement(hip, hand)

		return (displacement < HAND_HIP_THRESHOLD)
  
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
