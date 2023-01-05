from catching_judge import CatchingJudge
from point_projector import CameraIntrinsics

import sys

def main(input_filename):
	cam = CameraIntrinsics(
		focal_len=4.3,
		sensor_w=4.2,
		sensor_h=5.6,
		image_h=1960.0,
		image_w=1080.0,
	)

	with CatchingJudge(input_filename, cam) as judge:
		judge.process_and_write_video()

if __name__ == "__main__":
	main(sys.argv[1])