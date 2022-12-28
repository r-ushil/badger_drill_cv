class CatchingDrillContext():
	__ball_positions: list

	def __init__(self, cam_pose_estimator) -> None:
		self.__ball_positions = []
		self.__cam_pose_estimator = cam_pose_estimator

	def get_cam_pose_estimator(self):
		return self.__cam_pose_estimator

	def register_ball_position(self, ball_position):
		self.__ball_positions.append(ball_position)

	def get_ball_positions(self) -> list:
		return self.__ball_positions