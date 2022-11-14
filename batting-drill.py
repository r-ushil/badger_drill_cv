import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# setup pose estimation with min confidence for person detection / tracking
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5, model_complexity=2)

video = sys.argv[1]

cap = cv2.VideoCapture(video)

if cap.isOpened() == False:
    print("Error opening video file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# setup output video 
outdir, filename = video[:video.rfind('/')+1], video[video.rfind('/')+1:]

name, ext = filename.split('.')

out_filename = f'{outdir}{name}_annotated.{ext}'

out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
    'm', 'p', '4', 'v'), fps, (frame_width, frame_height))

# process per frame
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # convert colour format from BGR to RBG
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # run pose estimation on frame
    results = pose.process(image)

    # convert colour format back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # write pose landmarks from results onto frame
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    image = cv2.flip(image, 0)
    out.write(image)
    

pose.close()
cap.release()
out.release()


