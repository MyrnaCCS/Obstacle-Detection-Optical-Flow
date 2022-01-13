import numpy as np
import cv2
import random
from utils import project_points_with_homography

class GeometricTransformation(object):
	"""docstring for GeometricTransformation"""
	def __init__(self, tolerance):
		self.tol = tolerance
		self.homography = None

	def compute_planar_flow(self, points_old, points_current, u_optical):
		n = points_old.shape[0]

		# Auxiliares
		src_points = np.zeros((4, 2))
		dst_points = np.zeros((4, 2))
		u_planar = np.zeros((n, 2))
		u_planar_best = np.zeros((n, 2))

		# Resultado Homography
		H_best = np.zeros((3, 3))

		# Contadores
		count = 0
		max_inliers = -1

		# Iteraciones
		itr = int(np.log(1-0.99)/np.log(0.9375))

		while count < itr:
			# Select 3 random points
			for i in range(4):
				index = random.randint(0, n-1)
				src_points[i, 0] = points_old[index, 0, 0]
				src_points[i, 1] = points_old[index, 0, 1]
				dst_points[i, 0] = points_current[index, 0, 0]
				dst_points[i, 1] = points_current[index, 0, 1]

			# Estimate H
			H, mask = cv2.findHomography(src_points, dst_points)

			if H is None:
				count += 1
				continue

			# Compute the planar flow field
			u_planar = project_points_with_homography(points_old[:, 0, :], H)
			u_planar = u_planar - points_old[:, 0, :]

			# Match u_optical and u_planar
			euclidean_distance = u_optical[:, 0, :] - u_planar
			euclidean_distance = euclidean_distance * euclidean_distance
			euclidean_distance = np.sqrt(euclidean_distance[:, 0]+euclidean_distance[:, 1])

			# Count inliers
			count_inliers = (euclidean_distance <= self.tol).sum()

			# Save if is the best
			if count_inliers > max_inliers:
				max_inliers = count_inliers
				u_planar_best[:, :] = u_planar[:, :]
				H_best[:, :] = H[:, :]

			count += 1

		self.homography = H_best
		return u_planar_best

class AffineTransformation(GeometricTransformation):
	"""docstring for AffineTransformation"""
	def __init__(self, tolerance):
		super(AffineTransformation, self).__init__(tolerance)
	
	def compute_planar_flow(self, points_old, points_current, u_optical):
		n = points_old.shape[0]

		# Auxiliares
		D = np.zeros((6, 6))
		x_prime = np.zeros(6)
		u_planar = np.zeros((n, 2))
		u_planar_best = np.zeros((n, 2))

		# Resultado
		A = np.zeros((2, 2))
		b = np.zeros(2)

		# Contadores
		count = 0
		max_inliers = 0

		# Iteraciones
		itr = int(np.log(1-0.99)/np.log(0.784))

		while count < itr:
			# Select 3 random points
			for i in range(3):
				index = random.randint(0, n-1)
				D[2*i, 0] = D[2*i+1, 3] = points_old[index, 0, 0]
				D[2*i, 1] = D[2*i+1, 4] = points_old[index, 0, 1]
				D[2*i, 2] = D[2*i+1, 5] = 1.0
				x_prime[2*i] = points_current[index, 0, 0]
				x_prime[2*i+1] = points_current[index, 0, 1]

			# Estimate A and b
			D = D + 1e-6*np.eye(6)
			coef = np.dot(np.linalg.inv(D), x_prime)
			A[0, :] = coef[0:2]
			A[1, :] = coef[3:5]
			b[0] = coef[2]
			b[1] = coef[5]
			# Compute the planar flow field
			u_planar = np.dot(points_old[:, 0, :], np.transpose(A))
			u_planar[:, 0] += b[0]
			u_planar[:, 1] += b[1]
			u_planar = u_planar - points_old[:, 0, :]

			# Match u_optical and u_planar
			euclidean_distance = u_optical[:, 0, :] - u_planar
			euclidean_distance = euclidean_distance * euclidean_distance
			euclidean_distance = np.sqrt(euclidean_distance[:, 0]+euclidean_distance[:, 1])

			# Count inliers
			count_inliers = (euclidean_distance <= self.tol).sum()

			# Save if is the best
			if count_inliers > max_inliers:
				max_inliers = count_inliers
				u_planar_best[:, :] = u_planar[:, :]

			count += 1

		return u_planar_best