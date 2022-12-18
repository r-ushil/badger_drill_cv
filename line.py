import numpy as np

class Line:
	def __init__(self, p1, p2):
		assert p1.shape == (3, )
		assert p2.shape == (3, )
	
		self.d = p2 - p1
		self.v0 = p1

	
	# Finds the vector that rotates and scales 
	# v1 onto v2
	@staticmethod
	def find_rotation_matrix(v1, v2):
		# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
		assert v1.shape == (3, )
		assert v2.shape == (3, )

		v1norm = np.linalg.norm(v1)
		v2norm = np.linalg.norm(v2)

		v1unit = v1 / v1norm
		v2unit = v2 / v2norm

		v = np.cross(v1unit, v2unit)
		c = np.dot(v1unit, v2unit)
		I = np.identity(3)
		s = np.linalg.norm(v)
		k = v2norm / v1norm
		vx = np.array([
			[0, 	-v[2],	v[1]],
			[v[2], 	0,		-v[0]],
			[-v[1], v[0], 	0],
		])

		R = k * (I + vx + np.dot(vx, vx) * ((1 - c) / np.square(s)))

		assert np.linalg.norm(np.dot(R, v1) - v2) < 0.00001

		return R

	