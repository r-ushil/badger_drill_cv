import numpy as np

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
		calculated_d = np.dot(self.n, p) - self.d

		return calculated_d < TOLERANCE


	def sample_grid_points(self, grid_side_length, grid_spacing=1):
		# Ensure the grid side length is odd so it is centerable on self.p
		if grid_side_length % 2 == 0: grid_side_length += 1

		points = []
		for x in range(grid_side_length):
			for y in range(grid_side_length):
				v1_factor = x - (grid_side_length // 2)
				v2_factor = y - (grid_side_length // 2)

				point = self.p + (v1_factor * grid_spacing * self.v1)\
							   + (v2_factor * grid_spacing * self.v2)

				points.append(point)
		
		return points

def main():
	p1 = np.array([1, 2, 3])
	p2 = np.array([4, 9, 6])
	p3 = np.array([4, 8, 21])

	plane = Plane(p1, p2, p3)

	for p in plane.sample_grid_points(3, 1):
		print(plane.intersects_with_point(p))


if __name__ == "__main__":
	main()