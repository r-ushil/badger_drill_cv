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

		return np.dot(self.n, p) == self.d

def main():
	p1 = np.array([1, 2, 3])
	p2 = np.array([4, 9, 6])
	p3 = np.array([4, 8, 21])

	plane = Plane(p1, p2, p3)

	p4 = np.array([5, 6, 4])
	print(plane.intersects_with_point(p4))

if __name__ == "__main__":
	main()