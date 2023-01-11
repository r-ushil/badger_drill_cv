from typing import Optional
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc

class NotFoundError(Exception): pass

class Judge():
	fps: int
	frame_width: int
	frame_height: int

	video_capture: VideoCapture
	video_writer: VideoWriter

	def __init__(self, input_path: str, no_output: bool = False):
		reader = VideoCapture(input_path)

		if not reader.isOpened():
			raise NotFoundError("Error opening video file")

		should_output = not no_output

		frame_w = int(reader.get(3))
		frame_h = int(reader.get(4))
		fps = int(reader.get(5))

		writer_fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')

		out_path = Judge.__generate_output_video_path(input_path, no_output)
		writer = VideoWriter(out_path, writer_fourcc, fps, (frame_w, frame_h)) if should_output else None

		self.frame_width = frame_w
		self.frame_height = frame_h
		self.fps = fps

		self.video_capture = reader
		self.video_writer = writer

	@staticmethod
	def __generate_output_video_path(input_video_path: str, no_output: bool = False) -> Optional[str]:
		if no_output: return None

		output_video_directory, filename = \
			input_video_path[:input_video_path.rfind('/')+1], input_video_path[input_video_path.rfind('/')+1:]
  
		input_video_filename, input_video_extension = filename.split('.')
  
		output_video_path = f'{output_video_directory}{input_video_filename}_annotated.{input_video_extension}'
  
		return output_video_path

	def get_frames(self):
		while True:
			frame_present, frame = self.video_capture.read()

			if frame_present: yield frame
			else: break

	def write_frame(self, frame):
		if self.video_writer is not None:
			self.video_writer.write(frame)

	def get_video_dims(self) -> tuple[int, int]:
		return self.frame_width, self.frame_height

	def __enter__(self):
		return self

	def __exit__(self, type, value, traceback):
		self.video_capture.release()
		if self.video_writer is not None:
			self.video_writer.release()