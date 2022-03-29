import os
import cv2
from glob import glob

from lib.KeyPointsSelection import CornersPerRegion
from lib.GeometricTransformation import GeometricTransformation
from lib.KeyPointsClassification import OutliersPerRegion
from lib.ObstacleDetector import ObstacleDetector

if __name__ == "__main__":
	# Test in sequence of images
	paths = sorted(glob(os.path.join('images', '*' + '.png')))

	# Key points selection, key points classification and homography calculation
	key_points_selector = CornersPerRegion()

	max_outliers_tolerance = 4
	image_size = (90, 320)
	mesh_size = (2, 8)
	key_points_classificator = OutliersPerRegion(max_outliers_tolerance, image_size, mesh_size)

	tolerance = 1.0
	homography_calculator = GeometricTransformation(tolerance)
	optical_flow_detector = ObstacleDetector(image_size, tolerance, key_points_selector, homography_calculator, key_points_classificator)
	
	for i in range(len(paths)):
		img_old = cv2.imread(paths[2*i], 0)[150:, ...] # get bottom roi
		img_current = cv2.imread(paths[2*i+1], 0)[150:, ...] # get bottom roi
		# Bounding boxes of the obstacles
		bboxes = optical_flow_detector.compute_obstacles(img_old, img_current, str(i).zfill(5)+'.png')