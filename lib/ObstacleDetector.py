import numpy as np
import cv2
import sys
import os
from sklearn.cluster import MeanShift

from KeyPointsClassification import KeyPointsClassification
from ShowResults import *
sys.path.append('../')

class ObstacleDetector(object):
	"""docstring for ObstacleDetector"""
	def __init__(self, img_size, tolerance, corner_selection_strategy, geometric_transformation, corner_classification_strategy):
		self.rows = img_size[0]
		self.cols = img_size[1]
		self.corner_selection_strategy = corner_selection_strategy
		self.corner_classification_strategy = corner_classification_strategy
		self.geometric_transformation = geometric_transformation
		self.first_corner_classification_strategy = KeyPointsClassification(tolerance)

	def clustering_mean_shift(self, points_no_match, band_width):
		clustering = MeanShift(bandwidth=band_width).fit(points_no_match)
		unique_label = set(clustering.labels_)
		return clustering.labels_, unique_label

	def get_bounding_boxes(self, points, labels, unique_label, min_points):
		boxes = []
		for label in unique_label:
			cluster = points[labels == label]
			if len(cluster) >= min_points:
				x_min = max(0, min(cluster[:, 0]))
				y_min = max(0, min(cluster[:, 1]))
				x_max = min(self.cols-1, max(cluster[:, 0]))
				y_max = min(self.rows-1, max(cluster[:, 1]))
				boxes.append([x_min, y_min, x_max, y_max])
		return boxes

	def compute_obstacles(self, img_old, img_current, file_save_name):
		# Optical Flow
		points_old = self.corner_selection_strategy.define_key_points(img_old)
		points_current, st, err = cv2.calcOpticalFlowPyrLK(img_old, img_current, points_old, None, (21, 21), 1, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
		u_optical = points_current - points_old
		draw_optical_flow(img_old, points_old, points_current, file_save_name)
		
		# Planar flow
		u_planar = self.geometric_transformation.compute_planar_flow(points_old, points_current, u_optical)
		points_current_ground = u_planar + points_old[:, 0, :]
		draw_planar_flow(img_old, points_old, points_current_ground, file_save_name)

		# Clasifica puntos
		arg = {'optical_flow': u_optical,
				'planar_flow': u_planar,
				'key_points': points_old}
		no_match_points, match_points = self.first_corner_classification_strategy.classify_key_points(arg)

		# Clasificar por outliers per region
		f_arg = {'outliers': no_match_points,
				 'inliers': match_points}
		no_match, match = self.corner_classification_strategy.classify_key_points(f_arg)
		draw_points(img_old, no_match, match, file_save_name)

		if len(no_match) != 0:
			no_match = np.array(no_match) # from list to array
			# Clustering
			labels, unique_label = self.clustering_mean_shift(no_match, 50)
			# Draw clusters
			draw_clusters(img_old, no_match, labels, unique_label, file_save_name)
			# Draw boxes
			boxes = self.get_bounding_boxes(no_match, labels, unique_label,10)
			#Save image res boxes
			draw_box_obstacle(img_old, boxes, file_save_name)
			return boxes
		else:
			cv2.imwrite(os.path.join('clusters', file_save_name), img_old)
			cv2.imwrite(os.path.join('bboxes', file_save_name), img_old)
			return []