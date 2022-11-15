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
		mask = cv2.inRange(frame, (0, 0, 170), (20, 200, 185))

		contours, hierarchy = cv2.findContours(
		    image=mask,
		    mode=cv2.RETR_TREE,
		    method=cv2.CHAIN_APPROX_SIMPLE
		)

		mask = np.zeros(shape=mask.shape, dtype=np.uint8)
		mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

		cv2.drawContours(
		    image=mask,
		    contours=contours,
		    contourIdx=-1,
		    color=(0, 255, 0),
		    thickness=2,
		    lineType=cv2.LINE_AA
		)

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
