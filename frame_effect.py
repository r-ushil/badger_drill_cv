from enum import Enum
import numpy as np

class FrameEffectType(Enum):
	POINTS_2D_MULTIPLE = 1
	POINTS_3D_MULTIPLE = 2
	POINT_3D_SINGLE = 3
	KATCHET_FACE_POLY = 4
	TEXT = 5

class FrameEffect:
	def __init__(self, 
		frame_effect_type=None, 
		primary_label=None, 
		points_3d_multiple=None, 
		point_3d_single=None, 
		points_2d_multiple=None, 
		colour=None,
		display_label=None,
		show_label=None,
		katchet_face_poly=None,
		point_size=None
		):
		assert frame_effect_type is not None
		assert primary_label is not None
		self.frame_effect_type = frame_effect_type
		self.primary_label = primary_label

		match frame_effect_type:
			case FrameEffectType.POINTS_3D_MULTIPLE:
				assert points_3d_multiple is not None
				assert colour is not None
				assert show_label is not None

				self.points_3d_multiple = points_3d_multiple
				self.colour = colour
				self.show_label = show_label
				self.point_size = 10 if point_size is None else point_size

			case FrameEffectType.POINT_3D_SINGLE:
				assert point_3d_single is not None
				assert point_3d_single.shape == (3, 1)
				assert colour is not None
				assert show_label is not None

				if show_label and display_label is None:
					display_label = FrameEffect.generate_point_string(point_3d_single)

				self.point_3d_single = point_3d_single
				self.colour = colour
				self.display_label = display_label
				self.show_label = show_label

			case FrameEffectType.KATCHET_FACE_POLY:
				assert katchet_face_poly is not None
				assert colour is not None

				self.katchet_face_poly = katchet_face_poly
				self.colour = colour
			
			case FrameEffectType.TEXT:
				assert display_label is not None
				assert show_label is not None

				self.display_label = display_label
				self.show_label = show_label
			
			case FrameEffectType.POINTS_2D_MULTIPLE:
				assert points_2d_multiple is not None
				assert colour is not None
				assert show_label is not None

				self.points_2d_multiple = points_2d_multiple
				self.colour = colour
				self.show_label = show_label
				self.display_label = display_label

	
	@staticmethod
	def generate_point_string(point):
		assert point.shape == (3, ) or point.shape == (3, 1) or point.shape == (2, ) or point.shape == (2, 1)
		if point.shape == (3, ) or point.shape == (3, 1):
			if point.shape == (3, 1):
				point = point.reshape((3, ))

			wx = point[0]
			wy = point[1]
			wz = point[2]
		
			return f"({np.around(wx, 1)}, {np.around(wy, 1)}, {np.around(wz, 1)})"
		else:
			if point.shape == (2, 1):
				point = point.reshape((2, ))

			wx = point[0]
			wy = point[1]
		
			return f"({np.around(wx, 1)}, {np.around(wy, 1)}"




