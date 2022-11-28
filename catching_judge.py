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
		fps = int(self.video_capture.get(5))

		# setup output video 
		output_video_path = CatchingJudge.generate_output_video_path(input_video_path)

		self.video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
		'm', 'p', '4', 'v'), fps, (self.frame_width, self.frame_height))
  
	def process_and_write_video(self):
		frame_present, frame = self.video_capture.read()
		while frame_present:
			self.process_frame(frame)

			frame_present, frame = self.video_capture.read()
  
	def process_frame(self, frame):
		# convert colour format from BGR to RBG
		# gray_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)
		frame = cv2.flip(frame, -1)
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

		contour_lens = [(cv2.arcLength(contour, closed=False), contour) for contour in contours]

		# get the longest contour from the list
		contour_lens.sort(key=lambda x: x[0], reverse=True)

		# if no contours were found, return the original frame
		if len(contour_lens) == 0:
			return

		katchet_face_len = contour_lens[0][0]
		katchet_face = contour_lens[0][1]

		epsilon = 0.0125 * katchet_face_len
		katchet_face_poly = cv2.approxPolyDP(katchet_face, epsilon=epsilon, closed=True)

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

		cv2.circle(mask, (bottom_left_point[0], bottom_left_point[1]), 10, (0, 0, 255), -1)
		mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

		self.video_writer.write(mask)

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
		self.video_capture.release()
		self.video_writer.release()
