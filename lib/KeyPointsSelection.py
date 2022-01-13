import numpy as np
import cv2

class KeyPointsSelection(object):
	"""docstring for KeyPointsSelection"""
	def __init__(self, n_points=714):
		self.n_points = n_points

	def define_key_points(self, img_gray_old=None):
		# Puntos que se rastrean
		points_old = cv2.goodFeaturesToTrack(img_gray_old, mask=None, maxCorners=self.n_points, qualityLevel=1e-6, minDistance=3, 
											 blockSize=21, useHarrisDetector=1)
		return points_old

class CornersPerRegion(KeyPointsSelection):
	"""docstring for CornersPerRegion"""
	def __init__(self, n_points=714, n_rows=2, n_cols=8, maxPoints=70):
		super(CornersPerRegion, self).__init__(n_points)
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.maxPoints = maxPoints

	def define_key_points(self, img_gray_old=None):
		rows, cols = img_gray_old.shape
		roi_rows = rows//self.n_rows
		roi_cols = cols//self.n_cols
		points_old = np.zeros((self.maxPoints*self.n_rows*self.n_cols, 1, 2), dtype=np.float32)
		index = 0
		for i in range(self.n_rows):
			for j in range(self.n_cols):
				index_init_rows = i*roi_rows
				index_init_cols = j*roi_cols
				roi = img_gray_old[index_init_rows:index_init_rows+roi_rows, index_init_cols:index_init_cols+roi_cols]
				points_roi = cv2.goodFeaturesToTrack(roi, mask=None, maxCorners=self.maxPoints, qualityLevel=1e-9, minDistance=3, 
													 blockSize=21, useHarrisDetector=1)
				if points_roi is None:
					continue
				points_roi[..., 1] += index_init_rows
				points_roi[..., 0] += index_init_cols
				n = points_roi.shape[0]
				points_old[index:index+n, :, :] = points_roi[:, :, :]
				index += n
		return points_old[:index, ...]

class VerticesAtMesh(KeyPointsSelection):
	"""docstring for VerticesAtMesh"""
	def __init__(self, n_points=714, idx_init_rows=5, idx_init_cols=5, next_point_rows=6, next_point_cols=6, rows=90, cols=320):
		super(VerticesAtMesh, self).__init__(n_points)
		self.idx_init_rows = idx_init_rows
		self.idx_init_cols = idx_init_cols
		self.next_point_rows = next_point_rows
		self.next_point_cols = next_point_cols
		self.rows = rows
		self.cols = cols
		self.points_old = self.define_mesh_points()

	def define_mesh_points(self):
		# Puntos que se rastrean
		n_points_row = (self.rows-self.idx_init_rows-1)//self.next_point_rows 
		n_points_col = (self.cols-self.idx_init_cols-1)//self.next_point_cols
		points_old = np.zeros((n_points_row*n_points_col, 1, 2), dtype=np.float32)
		idx_final_rows = self.idx_init_rows + 1 + (n_points_row - 1) * self.next_point_rows
		idx_final_cols = self.idx_init_cols + 1 + (n_points_col - 1) * self.next_point_cols
		index_row = 0
		index_col = 0
		for i in range(self.idx_init_cols, idx_final_cols, self.next_point_cols):
			for j in range(self.idx_init_rows, idx_final_rows, self.next_point_rows):
				index_row = j//self.next_point_rows
				index_col = i//self.next_point_cols
				points_old[n_points_row*index_col+index_row, 0, 0] = i
				points_old[n_points_row*index_col+index_row, 0, 1] = j
		return points_old

	def define_key_points(self, img_gray_old=None):
		return self.points_old