import os
import cv2
from glob import glob

from lib.KeyPointsSelection import CornersPerRegion
from lib.GeometricTransformation import GeometricTransformation
from lib.KeyPointsClassification import OutliersPerRegion
from lib.ObstacleDetector import ObstacleDetector

if __name__ == "__main__":
	optical_flow_detector = ObstacleDetector((90, 320), 1.0, CornersPerRegion(), GeometricTransformation(1.0), OutliersPerRegion(4, (90, 320), (2, 8)))
	# Test in sequence of images
	paths = sorted(glob(os.path.join('images', '*' + '.png')))
	for i in range(50):
		img_old = cv2.imread(paths[2*i], 0)[150:, ...]
		img_current = cv2.imread(paths[2*i+1], 0)[150:, ...]
		index_str = str(i)
		bboxes = optical_flow_detector.compute_obstacles(img_old, img_current, index_str.zfill(5)+'.png')