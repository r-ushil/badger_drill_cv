import mediapipe as mp
import numpy as np
import cv2

SHOULDER_WIDTH_THRESHOLD = 0.05
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
		
		landmarks = self.generate_landmark_vectors(results)

		# write pose landmarks from results onto frame
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# TODO: - add logic to check that these landmarks are actually detected.
		# check for vertical alignment
		shoulder_width = self.feet_shoulder_width_apart(
			results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
			results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
			results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
			results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
		)

		image = cv2.flip(image, 0)
		cv2.putText(image, f'Shoulder Width: {shoulder_width}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

		self.video_writer.write(image)

	def generate_landmark_vectors(self, results):
		# iterate through detected landmarks, and add to list
		landmarks = []

		if results.pose_landmarks:
			for landmark in results.pose_landmarks.landmark:
				landmarks.append((
					int(landmark.x * self.frame_width),
					int(landmark.y * self.frame_height),
					int(landmark.z * self.frame_width)
				))

		return landmarks

	@staticmethod
	def calculate_x_displacement(a, b):
		return abs(a.x - b.x)

	# Returns a boolean on whether the feet are shoulder width apart
	def feet_shoulder_width_apart(self, left_shoulder, right_shoulder, left_foot, right_foot):

		# assume the feet aren't shoulder width apart if the landmarks aren't detected
		if (left_shoulder.visibility < 0.7 or right_shoulder.visibility < 0.7 or 
			left_foot.visibility < 0.7 or right_foot.visibility < 0.7):
			return False

		# calculate the x displacement between the feet and the shoulders
		shoulder_width = CoverDriveJudge.calculate_x_displacement(left_shoulder, right_shoulder)
		feet_width = CoverDriveJudge.calculate_x_displacement(left_foot, right_foot)

		return abs(shoulder_width - feet_width) < SHOULDER_WIDTH_THRESHOLD
  
	# Checks 3 joints are vertically aligned, with a tolerance on acceptable angle (in degrees)
	@staticmethod
	def check_vertical_alignment(shoulder, knee, foot, tolerance):
		vertical_alignment = CoverDriveJudge.calculate_angle(shoulder, knee, foot)
		return not (vertical_alignment > (180 - tolerance) and vertical_alignment < (180 + tolerance))

	# Calculates angles between 3 joints, given their 3d coordinates.
	@staticmethod
	def calculate_angle(a, b, c):
		a = np.array(a) # First
		b = np.array(b) # Mid
		c = np.array(c) # End
		# Calculate the angles between the vectors, in radians
		radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
		# Convert to degrees
		angle = np.abs(radians*180.0/np.pi)

		if angle > 180.0:
			angle = 360-angle

		return angle

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
