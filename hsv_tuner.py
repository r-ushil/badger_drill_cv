import cv2
import numpy as np
import sys

def setup_trackbars():
    # create window for trackbars
    cv2.namedWindow("image")

    # create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # nothing function for OpenCV
    def nothing(x):
        pass

    # set default value for HSV bars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

def tuner(input_filename):

    cap = cv2.VideoCapture(input_filename)

    # read frame
    frame_present, frame = cap.read()

    while frame_present:

        frame = cv2.flip(frame, -1)

        while True:

            # Display result image, cycle using 'n' key
            cv2.imshow('image', frame)
            if cv2.waitKey(10) & 0xFF == ord('n'):
                break

        frame_present, frame = cap.read()

    cv2.destroyAllWindows()


def main(input_filename):
    tuner(input_filename)


if __name__ == "__main__":
    main(sys.argv[1])
