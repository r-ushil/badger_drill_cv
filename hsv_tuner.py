import cv2
import numpy as np
import sys

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
