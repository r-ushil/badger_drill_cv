import cv2
import numpy as np
import sys

# Otto - Through the legs
# (hMin = 130 , sMin = 166, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)

# default values for the HSV trackbars
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

def print_changes():
  # print changes
  if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax)):
      print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
          hMin, sMin, vMin, hMax, sMax, vMax))
      phMin = hMin
      psMin = sMin
      pvMin = vMin
      phMax = hMax
      psMax = sMax
      pvMax = vMax

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

    setup_trackbars()

    # read frame
    frame_present, frame = cap.read()

    while frame_present:

        frame = cv2.flip(frame, -1)

        while True:

            # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            print_changes()

            # Display result image, cycle using 'n' key
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('n'):
                break

        frame_present, frame = cap.read()

    cv2.destroyAllWindows()


def main(input_filename):
    tuner(input_filename)


if __name__ == "__main__":
    main(sys.argv[1])
