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
		frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2GRAY)

		frame.flags.writeable = False
		# convert colour format back to BGR
		frame.flags.writeable = True

		frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
		self.video_writer.write(frame)

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
