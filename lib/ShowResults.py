import numpy as np
import cv2
import os
import random

def draw_optical_flow(img_old, points_old, points_current, file_save_name):
	n = points_old.shape[0]
	rows, cols = img_old.shape
	img_flujo_optico = np.zeros((rows, cols, 3), dtype=np.uint8)
	img_flujo_optico[..., 0] = img_flujo_optico[..., 1] = img_flujo_optico[..., 2] = img_old[...] 

	for i in range(n):
		x = points_old[i, 0, 0]
		y = points_old[i, 0, 1]
		x_ = points_current[i, 0, 0]
		y_ = points_current[i, 0, 1]
		cv2.line(img_flujo_optico, (x, y), (x_, y_), (194,227,80), 1)
		cv2.circle(img_flujo_optico, (x_, y_), 1, (194,227,80), 1)

	cv2.imwrite(os.path.join('opticalflow', file_save_name), img_flujo_optico)

def draw_planar_flow(img_old, points_old, points_current, file_save_name):
	n = points_old.shape[0]
	img_flujo_planar = np.zeros(img_old.shape+(3,), dtype=np.uint8)
	img_flujo_planar[..., 0] = img_flujo_planar[..., 1] = img_flujo_planar[..., 2] = img_old[...] 

	for i in range(n):
		x = points_old[i, 0, 0]
		y = points_old[i, 0, 1]
		x_ = int(points_current[i, 0])
		y_ = int(points_current[i, 1])
		cv2.line(img_flujo_planar, (x, y), (x_, y_), (134,233,184), 1)
		cv2.circle(img_flujo_planar, (x_, y_), 1, (134,233,184), 1)

	cv2.imwrite(os.path.join('planarflow', file_save_name), img_flujo_planar)

def draw_points(img_old, no_match_points, match_points, file_save_name):
	# Create a new output image
	img_class_points = np.zeros(img_old.shape+(3,), dtype=np.uint8)
	img_class_points[..., 0] = img_class_points[..., 1] = img_class_points[..., 2] = img_old[...]

	n = len(no_match_points)
	m = len(match_points)

	# Draw no match points
	for i in range(n):
		(x, y) = no_match_points[i]
		# Draw small circle
		cv2.circle(img_class_points, (int(x), int(y)), 2, (27,2,208), 1)
		
	# Draw match points
	for i in range(m):
		(x, y) = match_points[i]
		# Draw small circle
		cv2.circle(img_class_points, (int(x), int(y)), 1, (194,227,80), 1)

	cv2.imwrite(os.path.join('result', file_save_name), img_class_points)

def draw_clusters(img_old, points, labels, unique_label, file_save_name):
	img_clusters = np.zeros(img_old.shape+(3,), dtype=np.uint8)
	img_clusters[..., 0] = img_clusters[..., 1] = img_clusters[..., 2] = img_old[...]

	for label in unique_label:
		cluster = points[labels == label]
		b = random.randint(0, 255)
		g = random.randint(0, 255)
		r = random.randint(0, 255)
		for point in cluster:
			x = point[0]
			y = point[1]
			# Draw small circle
			cv2.circle(img_clusters, (int(x), int(y)), 2, (b,g,r), 1)

	cv2.imwrite(os.path.join('clusters', file_save_name), img_clusters)

def draw_box_obstacle(img_old, bounding_boxes, file_save_name):
	img_box_obs = np.zeros(img_old.shape+(3,), dtype=np.uint8)
	img_box_obs[..., 0] = img_box_obs[..., 1] = img_box_obs[..., 2] = img_old[...]
	for box in bounding_boxes:
		x_min = box[0]
		y_min = box[1]
		x_max = box[2]
		y_max = box[3]
		cv2.rectangle(img_box_obs, (x_min, y_min), (x_max, y_max), (27,2,208), 1)
	cv2.imwrite(os.path.join('bboxes', file_save_name), img_box_obs)