class CatchingDrillContext():
	__ball_positions: list

	def __init__(self) -> None:
		self.__ball_positions = []

	def register_ball_position(self, ball_position):
		self.__ball_positions.append(ball_position)

	def get_ball_positions(self) -> list:
		return self.__ball_positions