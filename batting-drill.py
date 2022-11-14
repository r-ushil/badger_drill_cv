import cv2
import mediapipe as mp
import numpy as np
import sys

def main():

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

    # setup output video 
    outdir, filename = video[:video.rfind('/')+1], video[video.rfind('/')+1:]

    name, ext = filename.split('.')

    out_filename = f'{outdir}{name}_annotated.{ext}'

    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), 30, (frame_width, frame_height))

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


# Calculates angles between 3 joints, given their 3d coordinates.
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

# Checks 3 joints are vertically aligned, with a tolerance on acceptable angle (in degrees)
def check_vertical_alignment(shoulder, knee, foot, tolerance):
    vertical_alignment = calculate_angle(shoulder, knee, foot)
    return not (vertical_alignment > (180 - tolerance) and vertical_alignment < (180 + tolerance))


if __name__ == '__main__':
    main()