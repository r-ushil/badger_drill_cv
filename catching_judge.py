import math
import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class CatchingJudge():

	def __init__(self, input_video_path):
		self.video_capture = cv2.VideoCapture(input_video_path)

		if not self.video_capture.isOpened():
			print("Error opening video file")
			raise TypeError

		self.frame_width = int(self.video_capture.get(3))
		self.frame_height = int(self.video_capture.get(4))
		self.fps = int(self.video_capture.get(5))

		self.pose_estimator = mp_pose.Pose(
			static_image_mode=False,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5,
			model_complexity=2,    
		)

		self.ball_positions = []

		# setup output video
		output_video_path = CatchingJudge.generate_output_video_path(
			input_video_path)

		self.video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
			'm', 'p', '4', 'v'), self.fps, (self.frame_width, self.frame_height))
	
	def process_and_write_video(self):
		frame_present, frame = self.video_capture.read()
		while frame_present:
			self.process_frame(frame)

			frame_present, frame = self.video_capture.read()

	def _resize(self, img):
		return cv2.resize(img, (375, 750))
	
	def detect_ball(self, frame):

		def find_bounce_point_frame(self):
			delta_x = abs(self.ball_positions[1][1][0] - self.ball_positions[0][1][0])
			delta_y = abs(self.ball_positions[1][1][1] - self.ball_positions[0][1][1])
			if delta_y < 0:
				return 1
			prev_angle = math.atan(delta_x/delta_y) 
			for f in range (2, len(self.ball_positions)):
				delta_x = self.ball_positions[f][1][0] - self.ball_positions[f-1][1][0]
				delta_y = self.ball_positions[f][1][1] - self.ball_positions[f-1][1][1]
				angle = math.atan(delta_x/delta_y)
				ratio = abs(angle/prev_angle)
				if delta_y > 0 & (ratio <= 1.1) & (ratio >= 0.9):
					prev_angle = angle
				else: 
					return f
			return -1

		def triangulate_bounce_point():
			f = find_bounce_point_frame(self)
			point_1 = self.ball_positions[f-2][1]
			point_2 = self.ball_positions[f-1][1]
			point_3 = self.ball_positions[f][1]
			point_4 = self.ball_positions[f+1][1]
			coeff_matrix = [[point_1[0] - point_2[0], point_4[0] - point_3[0]], [point_1[1] - point_2[1], point_4[1] - point_3[1]]]
			lambda_mu = np.matmul(np.linalg.inv(coeff_matrix), [[point_3[0] - point_2[0]],[point_3[1] - point_2[1]]])
			lambdaa = lambda_mu[0][0]
			return (round(point_2[0] + (lambdaa * (point_1[0]-point_2[0]))), round(point_2[1] + (lambdaa * (point_1[1]-point_2[1]))))



		# convert to HSV
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

		frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

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
			self.ball_positions.append(detected[0])
		
		print(len(self.ball_positions))
		if len(self.ball_positions) > 2: 
			f = find_bounce_point_frame(self) 
			print(f)
			if f != -1:
				for (area, centre, radius) in self.ball_positions:
					cv2.circle(frame, centre, radius, (0, 255, 0), cv2.FILLED)
				if (f < len(self.ball_positions) - 1):
					cv2.circle(frame, triangulate_bounce_point(), 20, (0, 0, 255), cv2.FILLED)
				return frame
		
		for (area, centre, radius) in self.ball_positions:
			cv2.circle(frame, centre, radius, (255, 0, 0), cv2.FILLED)

		# UNCOMMENT TO SHOW MASK FOR DEBUGGING
		# cv2.imshow('frame', self._resize(frame))
		# cv2.waitKey(1)

		return frame

		
	def detect_pose(self, frame):
		# Convert the BGR image to RGB before processing.
		results = self.pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		mp_drawing.draw_landmarks(
			frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

		# UNCOMMENT TO SHOW POSE DETECTION FOR DEBUGGING
		# cv2.imshow('MediaPipe Pose', self._resize(frame))
		# cv2.waitKey(1)

		return frame
	
	def katchet_board_detection(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		frame = cv2.GaussianBlur(frame, (9, 9), cv2.BORDER_DEFAULT)

		def h(val):
			return float(val) * (180.0 / 360.0)

		def s(val):
			return float(val) * (255.0 / 100.0)

		def v(val):
			return float(val) * (255.0 / 100.0)

		mask = cv2.inRange(frame,
									(h(0), s(30), v(80)),
									(h(30), s(100), v(100)))

		contours, hierarchy = cv2.findContours(
			image=mask,
			mode=cv2.RETR_TREE,
			method=cv2.CHAIN_APPROX_SIMPLE
		)

		mask = np.zeros(shape=mask.shape, dtype=np.uint8)
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

		contour_lens = [(cv2.arcLength(contour, closed=False), contour)
						for contour in contours]

		# get the longest contour from the list
		contour_lens.sort(key=lambda x: x[0], reverse=True)

		# if no contours were found, return the original frame
		if len(contour_lens) == 0:
			return

		katchet_face_len = contour_lens[0][0]
		katchet_face = contour_lens[0][1]

		epsilon = 0.0125 * katchet_face_len
		katchet_face_poly = cv2.approxPolyDP(
			katchet_face, epsilon=epsilon, closed=True)

		points = []
		for point in katchet_face_poly:
			points.append([point[0][0], point[0][1]])

		points.sort()
		bottom_left_point = points[0] if points[0][1] > points[1][1] else points[1]

		cv2.drawContours(
			image=mask,
			contours=[katchet_face_poly],
			contourIdx=0,
			color=(0, 0, 255),
			thickness=2,
			lineType=cv2.LINE_AA
		)

		cv2.circle(
			mask, (bottom_left_point[0], bottom_left_point[1]), 10, (0, 0, 255), -1)
		mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

	def process_frame(self, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.flip(frame, -1)

		# mask = self.katchet_board_detection(frame)
		ball_detected = self.detect_ball(frame)

		# run pose estimation on frame
		pose_detected = self.detect_pose(ball_detected)

		self.video_writer.write(pose_detected)

	@staticmethod
	def generate_output_video_path(input_video_path):
		output_video_directory, filename = \
			input_video_path[:input_video_path.rfind(
				'/')+1], input_video_path[input_video_path.rfind('/')+1:]

		input_video_filename, input_video_extension = filename.split('.')

		output_video_path = f'{output_video_directory}{input_video_filename}_annotated.{input_video_extension}'

		return output_video_path

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.video_capture.release()
		self.video_writer.release()
