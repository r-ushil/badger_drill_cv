from enum import Enum
import numpy as np

class FrameEffectType(Enum):
	POINTS_MULTIPLE = 1
	POINT_SINGLE = 2

class FrameEffect:
	def __init__(self, 
		frame_effect_type=None, 
		primary_label=None, 
		points_multiple=None, 
		point_single=None, 
		colour=None,
		display_label=None,
		show_label=None
		):
		assert frame_effect_type is not None
		assert primary_label is not None
		self.frame_effect_type = frame_effect_type
		self.primary_label = primary_label

		match frame_effect_type:
			case FrameEffectType.POINTS_MULTIPLE:
				assert points_multiple is not None
				assert colour is not None
				assert show_label is not None

				self.points_multiple = points_multiple
				self.colour = colour
				self.show_label = show_label

			case FrameEffectType.POINT_SINGLE:
				assert point_single is not None
				assert point_single.shape == (3, 1)
				assert colour is not None
				assert show_label is not None

				if show_label and display_label is None:
					display_label = FrameEffect.generate_point_string(point_single)

				self.point_single = point_single
				self.colour = colour
				self.display_label = display_label
				self.show_label = show_label
	
	@staticmethod
	def generate_point_string(point):
		assert point.shape == (3, ) or point.shape == (3, 1)

		if point.shape == (3, 1):
			point = point.reshape((3, ))

		wx = point[0]
		wy = point[1]
		wz = point[2]
		
		return f"({np.around(wx, 2)}, {np.around(wy, 2)}, {np.around(wz, 2)})"



