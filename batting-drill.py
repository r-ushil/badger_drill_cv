import cv2
import mediapipe as mp
import sys

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main(input_video_path):
  # setup pose estimation with min confidence for person detection / tracking
  pose_estimator = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                      min_tracking_confidence=0.5, model_complexity=2)

  video_capture = cv2.VideoCapture(input_video_path)

  if not video_capture.isOpened():
      print("Error opening video file")
      raise TypeError
  
  frame_width, frame_height, fps = get_video_metadata(video_capture)

  # setup output video 
  output_video_path = generate_output_video_path(input_video_path)
  video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
      'm', 'p', '4', 'v'), fps, (frame_width, frame_height))

  # process per frame
  while video_capture.isOpened():
      image_present, image = video_capture.read()
      if not image_present:
          break

      # convert colour format from BGR to RBG
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # run pose estimation on frame
      # TODO: make name more specific
      results = pose_estimator.process(image)

      # convert colour format back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # write pose landmarks from results onto frame
      mp_drawing.draw_landmarks(
          image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      image = cv2.flip(image, 0)
      video_writer.write(image)


  pose_estimator.close()
  video_capture.release()
  video_writer.release()

def generate_output_video_path(input_video_path):
  output_video_directory, filename = \
    input_video_path[:input_video_path.rfind('/')+1], input_video_path[input_video_path.rfind('/')+1:]

  input_video_filename, input_video_extension = filename.split('.')

  output_video_path = f'{output_video_directory}{input_video_filename}_annotated.{input_video_extension}'

  return output_video_path

def get_video_metadata(video_capture):
  frame_width = int(video_capture.get(3))
  frame_height = int(video_capture.get(4))
  fps = int(video_capture.get(5))

  return (frame_width, frame_height, fps)


if __name__ == "__main__":
  main(sys.argv[1])

