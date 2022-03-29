import numpy as np
import cv2

class KeyPointsSelection(object):
	"""This class returns n key points selected by Harris algorithm"""
	def __init__(self, n_key_points=714):
		self.n_key_points = n_key_points

	def define_key_points(self, img_gray=None):
		key_points = cv2.goodFeaturesToTrack(img_gray, mask=None, maxCorners=self.n_key_points, qualityLevel=1e-6, minDistance=3, blockSize=11, useHarrisDetector=1)
		return key_points


class CornersPerRegion(KeyPointsSelection):
	"""This class returns the sum of the key points selected by Harris algorithm in every region in the image"""
	def __init__(self, n_key_points=714, roi_rows=45, roi_cols=40, max_points=70):
		super(CornersPerRegion, self).__init__(n_key_points)
		self.roi_rows = roi_rows # Region size
		self.roi_cols = roi_cols # Region size
		self.max_points = max_points # Max key points per region

	def define_key_points(self, img_gray=None):
		rows, cols = img_gray.shape
		index_init_rows = 0
		index_init_cols = 0
		while (index_init_rows < rows):
			roi = img_gray[index_init_rows:index_init_rows+self.roi_rows, index_init_cols:index_init_cols+self.roi_cols]
			key_points_roi = cv2.goodFeaturesToTrack(roi, mask=None, maxCorners=self.max_points, qualityLevel=1e-9, minDistance=3, blockSize=11, useHarrisDetector=1)
			if key_points_roi is None:
				continue
			if index_init_rows == 0 and index_init_cols == 0:
				key_points = key_points_roi
			else:
				key_points_roi[..., 1] += index_init_rows
				key_points_roi[..., 0] += index_init_cols
				key_points = np.concatenate([key_points, key_points_roi])
			if index_init_cols < cols - self.roi_cols:
				index_init_cols += self.roi_cols
			else:
				index_init_cols = 0
				index_init_rows += self.roi_rows
		return key_points


class VerticesAtMesh(KeyPointsSelection):
	"""This class returns n key points selected as vertices in a mesh with n_rows x n_cols cells"""
	def __init__(self, n_rows=15, n_cols=52):
		super(VerticesAtMesh, self).__init__((n_rows - 1) * (n_cols-1))
		self.n_rows = n_rows
		self.n_cols = n_cols

	def define_key_points(self, img_gray=None):
		rows, cols = img_gray.shape
		next_point_rows = rows // self.n_rows
		next_point_cols = cols // self.n_cols
		key_points = np.zeros((self.n_key_points, 1, 2), dtype=np.float32)
		for row in range(self.n_rows-1):
			for col in range(self.n_cols-1):
				key_points[(self.n_cols-1)*row+col, 0, 0] = col * next_point_cols + next_point_cols
				key_points[(self.n_cols-1)*row+col, 0, 1] = row * next_point_rows + next_point_rows
		return key_points