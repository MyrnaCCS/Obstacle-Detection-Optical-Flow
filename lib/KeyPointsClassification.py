import numpy as np
from utils import *

class KeyPointsClassification(object):
	"""docstring for KeyPointsClassification"""
	def __init__(self, tolerance):
		self.tol = tolerance

	def classify_key_points(self, arg):
		u_optical = arg['optical_flow']
		u_planar = arg['planar_flow']
		points_old = arg['key_points']
		# Numero de puntos a evaluar
		n = points_old.shape[0]
		# Resultado
		no_match_points = []
		match_points = []
		# Calcula norma 2 entre flujo optico y flujo planar
		euclidean_distance = u_optical[:, 0, :] - u_planar
		euclidean_distance = euclidean_distance * euclidean_distance
		euclidean_distance = np.sqrt(euclidean_distance[:, 0]+euclidean_distance[:, 1])
		# Clasifica en: match (no obstaculos) y no_match (obstaculos)
		for i in range(n):
			if euclidean_distance[i] <= self.tol:
				match_points.append(points_old[i, 0, :])
			else:
				no_match_points.append(points_old[i, 0, :])
		return no_match_points, match_points

class OutliersPerRegion(KeyPointsClassification):
	"""docstring for OutliersPerRegion"""
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
		roi_rows = self.img_rows//self.rows
		roi_cols = self.img_cols//self.cols
		# Solo analizamos outliers
		n = len(no_match_points) 
		# How many outlier per region?
		counter = count_outliers(no_match_points, self.img_rows, self.img_cols, self.rows, self.cols)
		# Result
		no_match = []
		# Classify at: match (ground plane) & no_match (obstacles)
		for i in range(n):
			col, row = no_match_points[i]
			# a cual region pertenece el outlier
			index = self.cols*(int(row)//roi_rows)+(int(col)//roi_cols)
			count = counter[index]
			# si hay suficientes outliers, entonces el punto se queda en no_match
			if count >= self.tol:
				no_match.append(no_match_points[i])
			else:
				match_points.append(no_match_points[i])
		return no_match, match_points

class Correlation(KeyPointsClassification):
	"""docstring for Correlation"""
	def __init__(self, tolerance, patch_size):
		super(Correlation, self).__init__(tolerance)
		self.patch_size = patch_size

	def classify_key_points(self, arg):
		no_match_points = arg['outliers']
		match_points = arg['inliers']
		img_old = arg['img_old']
		img_current = arg['img_current']
		H = arg['homography']
		# Compute correlation to all key points
		correlation = compute_correlation(no_match_points, img_old, img_current, self.patch_size, H)
		# Numero de puntos
		n = correlation.shape[0]
		# Resultado
		no_match = []
		# Classify 
		for i in range(n):
			if correlation[i] <= self.tol:
				match_points.append(no_match_points[i])
			else:
				no_match.append(no_match_points[i])
		return no_match, match_points