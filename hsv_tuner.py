import cv2
import numpy as np
import sys
from enum import Enum

# Otto - Through the legs
# (hMin = 130 , sMin = 166, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)

# default values for the HSV trackbars

class Params(Enum):
    hMin = 0
    sMin = 1
    vMin = 2
    hMax = 3
    sMax = 4
    vMax = 5

def resize(img):
    return cv2.resize(img, (375, 750))

def print_changes(prev, curr):
  # print changes

  if not np.array_equal(prev, curr):
    print('HSV values changed:', curr)
    return curr.copy()
  
  return prev

def setup_trackbars():
    # create window for trackbars
    cv2.namedWindow("image")

    # nothing function for OpenCV
    def nothing(x):
        pass

    # create trackbars for color change
    cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

    # set default value for HSV bars
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

def tuner(input_filename):

    cap = cv2.VideoCapture(input_filename)

    setup_trackbars()

    curr = np.zeros(len(Params))
    prev = np.zeros(len(Params))

    # read frame
    frame_present, frame = cap.read()

    while frame_present:

        frame = cv2.flip(frame, -1)

        while True:

            # get current positions of all trackbars
            curr[Params.hMin.value] = cv2.getTrackbarPos('HMin', 'image')
            curr[Params.sMin.value] = cv2.getTrackbarPos('SMin', 'image')
            curr[Params.vMin.value] = cv2.getTrackbarPos('VMin', 'image')
            curr[Params.hMax.value] = cv2.getTrackbarPos('HMax', 'image')
            curr[Params.sMax.value] = cv2.getTrackbarPos('SMax', 'image')
            curr[Params.vMax.value] = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and max HSV values to display
            lower = curr[:3]
            upper = curr[3:]

            # convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            prev = print_changes(prev, curr)

            # Display result image, cycle using 'n' key
            cv2.imshow('image', resize(result))
            if cv2.waitKey(10) & 0xFF == ord('n'):
                break

        frame_present, frame = cap.read()

    cv2.destroyAllWindows()


def main(input_filename):
    tuner(input_filename)


if __name__ == "__main__":
    main(sys.argv[1])
