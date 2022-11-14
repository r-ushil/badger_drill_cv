import mediapipe as mp
import cv2

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

    frame_width, frame_height, fps = self.get_video_metadata(self.video_capture)

    # setup output video 
    output_video_path = self.generate_output_video_path(input_video_path)

    self.video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (frame_width, frame_height))
  
  def process_and_write_video(self):
    image_present, image = self.video_capture.read()
    while image_present:
        self.process_frame(image)

        image_present, image = self.video_capture.read()
  
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
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    image = cv2.flip(image, 0)
    self.video_writer.write(image)
  
  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.pose_estimator.close()
    self.video_capture.release()
    self.video_writer.release()
    
  @staticmethod
  def get_video_metadata(video_capture):
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = int(video_capture.get(5))

    return (frame_width, frame_height, fps)

  @staticmethod
  def generate_output_video_path(input_video_path):
    output_video_directory, filename = \
      input_video_path[:input_video_path.rfind('/')+1], input_video_path[input_video_path.rfind('/')+1:]
  
    input_video_filename, input_video_extension = filename.split('.')
  
    output_video_path = f'{output_video_directory}{input_video_filename}_annotated.{input_video_extension}'
  
    return output_video_path



