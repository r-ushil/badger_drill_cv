import numpy as np
from numpy.testing import assert_array_almost_equal

class Plane:
	def __init__(self, p1, p2, p3):
		assert p1.shape == (3, )
		assert p2.shape == (3, )
		assert p3.shape == (3, )

		self.p = p1

		# Get two vectors along the plane
		self.v1 = (p2 - p1) / np.linalg.norm(p2 - p1)
		self.v2 = (p3 - p1) / np.linalg.norm(p3 - p1)

		# Calculate normal vector d, store as plane representation:
		# ax + by + cz = d
		self.n = np.cross(self.v1, self.v2)
		self.d = np.dot(self.n, p1)

	def intersects_with_point(self, p):
		assert p.shape == (3, )

		TOLERANCE = 0.000001
		calculated_d_diff = np.dot(self.n, p) - self.d

		return calculated_d_diff < TOLERANCE


	# Returns a grid of points centred at the first point the plane was
	# initialized from.
	# Spaced by grid_spacing, containing grid_side_length^2 points.
	def sample_grid_points(self, grid_side_length, grid_spacing=1):

		# Ensure the grid side length is odd so it is centerable on self.p
		if grid_side_length % 2 == 0: grid_side_length += 1

		points = []
		for x in range(grid_side_length):
			for y in range(grid_side_length):
				v1_factor = x - (grid_side_length // 2)
				v2_factor = y - (grid_side_length // 2)

				point = self.p\
					+ (v1_factor * grid_spacing * self.v1)\
					+ (v2_factor * grid_spacing * self.v2)

				points.append(point)
		
		return points
	
	# Returns the angle between two planes in radians
	def calculate_angle_with_plane(self, plane):
		# cos theta = (n1 . n2) / (|n1| * |n2|)

		# Assert that the planes are not parallel
		assert np.linalg.norm(self.n - plane.n) >= 0.0001

		return np.arccos(
			np.dot(self.n, plane.n) / 
			(np.linalg.norm(self.n) * np.linalg.norm(plane.n))
		)
	
	def calculate_intersection_point_between_planes(self, plane):
		# Assumes both planes are vertical
		# Assumes both planes intersect
		# Assumes Z = 0 for the found point the lies on the line that intersects both planes

		(a1, b1, _) = self.n
		(a2, b2, _) = plane.n

		mat = np.array([
			[a1, b1],
			[a2, b2]
		])

		# Find X and Y coordinates of the point
		res = np.linalg.inv(mat) @ np.array([self.d, plane.d])

		# Add the Z coordinate of 0 to the point
		res = np.concatenate((res, np.array([0.], dtype=np.float64)), axis=0)

		return res
	
	@staticmethod
	def multiply_orthogonal_matrix_by_non_orthogonal_vec(mat, vec):
		assert mat.shape == (4, 4)
		assert vec.shape == (3, 1)

		res = mat @ np.concatenate((vec, np.array([[1.]])), axis=0)
		res = np.delete(res, 3, axis=0)

		return res

	@staticmethod
	def get_rotation_matrix_about_point(theta_rad, point, axis = "Z"):
		assert point.shape == (3, )

		translation_to_origin = np.array([
			[1, 0, 0, -point[0]],
			[0, 1, 0, -point[1]],
			[0, 0, 1, -point[2]],
			[0, 0, 0, 1		   ],
		])

		translation_back = np.array([
			[1, 0, 0, point[0]],
			[0, 1, 0, point[1]],
			[0, 0, 1, point[2]],
			[0, 0, 0, 1		  ],
		])

		# TODO: Switch to use Y-axis as vertical when we migrate
		c = np.cos(theta_rad)
		s = np.sin(theta_rad)

		rotation = None
		if axis == "Z":
			rotation = np.array([
				[c, -s, 0, 0],
				[s, c,  0, 0],
				[0, 0,  1, 0],
				[0, 0,  0, 1],
			])
		elif axis == "Y":
			rotation = np.array([
				[c,  0, s, 0],
				[0,  1, 0, 0],
				[-s, 0, c, 0],
				[0,  0, 0, 1],
			])

		return translation_back @ (rotation @ translation_to_origin)


def main():
	p1 = np.array([1, 2, 3])
	p2 = np.array([4, 9, 6])
	p3 = np.array([4, 8, 21])

	plane = Plane(p1, p2, p3)

	for p in plane.sample_grid_points(3, 1):
		assert plane.intersects_with_point(p)
	
	R1 = Plane.get_rotation_matrix_about_point(7, np.array([1, 2, 3]), "Y")
	R2 = Plane.get_rotation_matrix_about_point(-7, np.array([1, 2, 3]), "Y")
	R3 = Plane.get_rotation_matrix_about_point(4, np.array([4, 5, 6]), "Z")
	R4 = Plane.get_rotation_matrix_about_point(-4, np.array([4, 5, 6]), "Z")

	assert_array_almost_equal(np.identity(4), R4 @ (R3 @ (R2 @ R1)))

	plane2 = Plane(
		np.array([0, 0, 0]),
		np.array([1, 0, 0]),
		np.array([0, 0, 1])
	)

	plane3 = Plane(
		np.array([0, 0, 0]),
		np.array([0, 1, 0]),
		np.array([0, 0, 1])
	)

	res = plane2.calculate_intersection_point_between_planes(plane3)
	assert_array_almost_equal(res, np.array([0, 0, 0]))

	affine = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [0, 0, 0, 1],
    ])

	vec = np.array([
		[1],
		[2],
		[3]
	])

	res = Plane.multiply_orthogonal_matrix_by_non_orthogonal_vec(affine, vec)

	print(res)


if __name__ == "__main__":
	main()