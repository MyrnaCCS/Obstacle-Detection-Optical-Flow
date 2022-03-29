import numpy as np

class PositionControl(object):
	"""docstring for PositionControl"""
	def __init__(self, position_ref=None):
		self.position_ref = position_ref

	def set_reference_position(self, position_ref):
		self.position_ref = position_ref

	def compute_error(self, position_current):
		error = np.linalg.norm(self.position_ref - position_current)
		return error

	def compute_velocities(self, position_current):
		# Angulo en Z
		tz = position_current[2]
		# Calcula error
		error = self.position_ref - position_current
		# Calcula matriz de rotacion
		R_z = np.array([[np.cos(tz), np.sin(tz), 0.], [-np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]])
		# Transforma a sistema de robot humanoide
		v = np.dot(R_z, np.array([error[0], error[1], 1]))
		# Delimita velocidades
		v_max = 0.7
		t_max = 0.09
		# Multiplica por ganancias
		v_x = max(-v_max, min(v_max, 0.6 * v[0]))
		v_y = max(-v_max, min(v_max, 0.6 * v[1]))
		# Verifica error en orientacion
		if error[2] < - 3.14:
			error[2] = 6.28 + error[2]
		if error[2] > 3.14:
			error[2] = - 6.28 - error[2]
		theta_z = max(-t_max, min(t_max, 0.3 * error[2]))
		return v_x, v_y, theta_z