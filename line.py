class Line:
	def __init__(self, p1, p2):
		assert p1.shape == (3, )
		assert p2.shape == (3, )
	
		self.d = p2 - p1
		self.v0 = p1
	