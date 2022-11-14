import mediapipe as mp
import numpy as np
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

    self.frame_width = frame_width
    self.frame_height = frame_height
    self.fps = fps

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

    # iterate through detected landmarks, and add to list
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((
              int(landmark.x * self.frame_width), \
              int(landmark.y * self.frame_height), \
              int(landmark.z * self.frame_width)
              ))

    # write pose landmarks from results onto frame
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # TODO: - add logic to check that these landmarks are actually detected.
    # check for vertical alignment
    aligned = self.check_vertical_alignment(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        10
    )

    cv2.putText(image, f'Aligned: {aligned}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    image = cv2.flip(image, 0)
    self.video_writer.write(image)
  
  def __enter__(self):
    return self

  def __exit__(self, type, value, traceback):
    self.pose_estimator.close()
    self.video_capture.release()
    self.video_writer.release()
  
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



