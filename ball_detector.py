from enum import Enum
from typing import Optional
from augmented_frame import AugmentedFrame
from cv2 import arcLength, \
	bitwise_and, \
	bitwise_not, \
	bitwise_or, \
	contourArea, \
	minEnclosingCircle, \
	findContours, \
	inRange, \
	morphologyEx, \
	Mat, \
	CHAIN_APPROX_SIMPLE, \
	MORPH_ERODE, \
	MORPH_DILATE, \
	RETR_EXTERNAL
from numpy import array, isfinite, nan, ones, pi, uint8
from pandas import DataFrame

class BallDetector():
	__measured_ball_positions: list
	__interpol_ball_positions: Optional[list]

	__mask_prev_frame: Optional[Mat]

	def __init__(self) -> None:
		self.__measured_ball_positions = []
		self.__interpol_ball_positions = None
		self.__mask_prev_frame = None

	def process(self, augmented_frame: AugmentedFrame):
		frame = augmented_frame.frame_hsv()

		# define ranges of red colors in HSV
		# note: there are two ranges since the reds wrap around at 180.
		lower_red_1 = array([170, 110, 80])
		upper_red_1 = array([180, 255, 180])

		lower_red_2 = array([0, 110, 80])
		lower_red_2 = array([5, 255, 180])

		# Threshold the HSV image to get only red colours
		mask_1 = inRange(frame, lower_red_1, upper_red_1)
		mask_2 = inRange(frame, lower_red_2, lower_red_2)

		# Combine both masks to get complete data
		mask = bitwise_or(mask_1, mask_2)
		mask_curr_frame = mask
		mask_prev_frame = self.__mask_prev_frame

		# use morphology to remove noise
		kernel = ones((3, 3), uint8)

		if mask_prev_frame is not None:
			# Restrict detections to those in new areas
			undetected_area_mask = bitwise_not(mask_prev_frame)
			mask = bitwise_and(mask, undetected_area_mask)

		mask = morphologyEx(mask, MORPH_ERODE, kernel, iterations=1)
		mask = morphologyEx(mask, MORPH_DILATE, kernel, iterations=3)

		# Store dilated mask
		self.__mask_prev_frame = morphologyEx(mask_curr_frame, MORPH_DILATE, kernel, iterations=3)

		# find the circle blobs in the mask
		contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

		detected = []

		for _, c in enumerate(contours):
			# get circle area
			area = contourArea(c)

			# get circle perimeter
			perimeter = arcLength(c, True)

			# get circlularity
			circularity = 4 * pi * (area / (perimeter * perimeter))

			min_circularity = 0.5
			min_area = 100

			(x, y), radius = minEnclosingCircle(c)

			centre = (int(x), int(y))
			radius = int(radius)

			# add blob information if exceeds thresholds
			if circularity > min_circularity and area > min_area:
				detected.append((area, centre, radius))

		# Check whether a ball was detected
		if len(detected) > 0:
			# Pick detection with maximum area
			(_, (x, y), _) = max(detected, key=lambda x: x[0])

			# Add ball position and mark as not interpolated
			self.__measured_ball_positions.append((x, y, False))
		else:
			# Add dummy position for frame and mark to be interpolated
			self.__measured_ball_positions.append((nan, nan, True))

	def interpolate_ball_positions(self):
		# Construct DataFrame of ball positions and whether they are interpolated.
		measured_df = DataFrame(self.__measured_ball_positions)

		# Interpolate ball positions using cubic spline interpolation.
		# Note: does not extrapolate beyond the final measurement since ball could be dead or caught.
		interpol_df = measured_df.interpolate(method="cubicspline", limit_direction='forward', limit_area='inside')
		interpol_iter = interpol_df.itertuples(index=False, name=None)

		# Reconstruct ball positions to match data structure of `ball_positions`.
		interpol_ball_positions = [(int(x), int(y), bool(isinterpol)) if isfinite(x) and isfinite(y) else None for (x, y, isinterpol) in interpol_iter]

		self.__interpol_ball_positions = interpol_ball_positions

	def get_ball_positions(self):
		assert self.__interpol_ball_positions is not None

		return self.__interpol_ball_positions

class CriticalPointType(Enum):
	BOUNCE = 1
	CATCH = 2
	DROP = 3

class CriticalPoint():
	def __init__(self,point_type: CriticalPointType, frame_num: int, position_2d: tuple[int, int]) -> None:
		self.__point_type = point_type
		self.__frame_num = frame_num
		self.__position_2d = position_2d

	def get_type(self):
		return self.__point_type

	def get_frame_num(self):
		return self.__frame_num

	def get_position_2d(self):
		return self.__position_2d

class CriticalBallPointDetector():
	def __init__(self, ball_positions, left_hand_poses_2d, right_hand_poses_2d):
		self.__ball_positions = ball_positions
		self.__left_hand_poses_2d = left_hand_poses_2d
		self.__right_hand_poses_2d = right_hand_poses_2d

	def get_critical_points(self):
		from more_itertools import windowed
		from numpy import arccos, clip, dot, pi
		from numpy.linalg import norm

		def unit(vec):
			return vec / norm(vec)

		found_bounce = False
		frame_num_bounce = None

		# Use window size 4 and then extrapolate down and find intersection point of two line segments either side
		for (frame_num, (bp_fst, bp_snd, bp_thd)) in enumerate(windowed(self.__ball_positions, 3)):
			if bp_fst is None or bp_snd is None or bp_thd is None:
				continue

			(bp_fst_x, bp_fst_y, _) = bp_fst
			(bp_snd_x, bp_snd_y, _) = bp_snd
			(bp_thd_x, bp_thd_y, _) = bp_thd

			delta_fst = array([bp_fst_x - bp_snd_x, bp_fst_y - bp_snd_y])
			delta_snd = array([bp_snd_x - bp_thd_x, bp_snd_y - bp_thd_y])

			delta_fst = unit(delta_fst)
			delta_snd = unit(delta_snd)

			angle = arccos(clip(dot(delta_fst, delta_snd), -1.0, 1.0))

			if angle > pi / 4:
				frame_num_bounce = frame_num
				yield CriticalPoint(CriticalPointType.BOUNCE, frame_num + 1, (bp_snd_x, bp_snd_y))
				break

		if frame_num_bounce is None:
			# Ball did not bounce so no critical points
			return

		for (frame_num, (ball_pos, left_hand_pose_2d, right_hand_pose_2d)) in enumerate(zip(
			self.__ball_positions,
			self.__left_hand_poses_2d,
			self.__right_hand_poses_2d
		)):
			if frame_num < frame_num_bounce:
				# The ball cannot be caught before its bounced.
				continue

			if ball_pos is None:
				# The ball position has not been detected.
				continue

			if left_hand_pose_2d is None or right_hand_pose_2d is None:
				# Either the right hand and the left hand have not been detected, so skip
				continue

			(ball_x, ball_y, _) = ball_pos

			[l_hand_x, l_hand_y] = left_hand_pose_2d
			[r_hand_x, r_hand_y] = right_hand_pose_2d

			ball_to_r_hand = array([ball_x - r_hand_x, ball_y - r_hand_y])
			ball_to_l_hand = array([ball_x - l_hand_x, ball_y - l_hand_y])

			THRESHOLD = 30

			if norm(ball_to_r_hand) < THRESHOLD or norm(ball_to_l_hand) < THRESHOLD:
				yield CriticalPoint(CriticalPointType.CATCH, frame_num, (ball_x, ball_y))
