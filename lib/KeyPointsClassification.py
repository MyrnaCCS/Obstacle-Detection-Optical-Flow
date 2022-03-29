import numpy as np
from utils import *

class KeyPointsClassification(object):
	"""This class returns two list: 1) obstacle key points, 2) ground plane key points.
	   
	   1) The key point belongs to an obstacle if the euclidean error is larger than the tolerance.

	   2) The key point belongs to the ground plane if the error is less than the tolerance.

	"""
	def __init__(self, tolerance):
		self.tol = tolerance

	def classify_key_points(self, arg):
		u_optical = arg['optical_flow']
		u_planar = arg['planar_flow']
		key_points = arg['key_points']
		# Result
		no_match_points = []
		match_points = []
		# Compute euclidean error
		euclidean_distance = np.linalg.norm(u_optical[:, 0, ...] - u_planar, axis=1)
		# Classify: obstacle key points or ground plane key points
		for i in range(key_points.shape[0]):
			if euclidean_distance[i] <= self.tol:
				match_points.append(key_points[i, 0, ...])
			else:
				no_match_points.append(key_points[i, 0, ...])
		return no_match_points, match_points

class OutliersPerRegion(KeyPointsClassification):
	"""This class returns two list: 1) obstacle key points, 2) ground plane key points.

	   1) The key point belongs to an obstacle if the number of outliers is larger than tolerance.
	   
	   2) The key point belongs to the ground plane if the number of outliers is less than tolerance.
	
	"""
	def __init__(self, tolerance, image_size, mesh_size):
		super(OutliersPerRegion, self).__init__(tolerance)
		self.img_rows = image_size[0]
		self.img_cols = image_size[1]
		self.rows = mesh_size[0]
		self.cols = mesh_size[1]

	def classify_key_points(self, arg):
		no_match_points = arg['outliers']
		match_points = arg['inliers']
		# Size of region
		roi_rows = self.img_rows // self.rows
		roi_cols = self.img_cols // self.cols
		# How many outlier per region?
		counter = count_outliers(no_match_points, self.img_rows, self.img_cols, self.rows, self.cols)
		# Result
		no_match = []
		# Classify: 1) match (ground plane), 2) no_match (obstacles)
		for i in range(len(no_match_points)):
			col, row = no_match_points[i]
			# a cual region pertenece el outlier
			idx_region = self.cols * (int(row) // roi_rows) + (int(col) // roi_cols)
			n_outliers = counter[idx_region]
			# si hay suficientes outliers, entonces el punto se queda en no_match
			if n_outliers >= self.tol:
				no_match.append(no_match_points[i])
			else:
				match_points.append(no_match_points[i])
		return no_match, match_points


class Correlation(KeyPointsClassification):
	"""This class classify outliers: 1) obstacle key points, 2) ground plane key points.
	   
	   1) The key point belongs to an obstacle if the correlation between two patches is less than the tolerance.

	   2) The key point belongs to the ground plane if the correlation is larger than the tolerance.

	"""
	def __init__(self, tolerance, patch_size):
		super(Correlation, self).__init__(tolerance)
		self.patch_size = patch_size

	def classify_key_points(self, arg):
		no_match_points = arg['outliers']
		match_points = arg['inliers']
		img_old = arg['img_old']
		img_current = arg['img_current']
		H = arg['homography']
		# Compute key points correlation
		correlation = compute_correlation(no_match_points, img_old, img_current, self.patch_size, H)
		# Resultado
		no_match = []
		# Classify 
		for i in range(correlation.shape[0]):
			if correlation[i] <= self.tol:
				match_points.append(no_match_points[i])
			else:
				no_match.append(no_match_points[i])
		return no_match, match_points