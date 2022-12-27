from enum import Enum
import numpy as np

class FrameEffectType(Enum):
	POINTS_MULTIPLE = 1
	POINT_SINGLE = 2

class FrameEffect:
	def __init__(self, 
		frame_effect_type, 
		points_multiple=None, 
		point_single=None, 
		colour=None,
		label=None,
		show_label=None
		):
		assert frame_effect_type is not None
		self.frame_effect_type = frame_effect_type

		match frame_effect_type:
			case FrameEffectType.POINTS_MULTIPLE:
				assert points_multiple is not None
				assert colour is not None

				self.points_multiple = points_multiple
				self.colour = colour

			case FrameEffectType.POINT_SINGLE:
				assert point_single is not None
				assert point_single.shape == (3, 1)
				assert colour is not None
				assert label is not None
				assert show_label is not None

				self.point_single = point_single
				self.colour = colour
				self.label = label
				self.show_label = show_label


