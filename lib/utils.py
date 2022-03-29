import numpy as np
import cv2

def count_outliers(no_match_points, img_rows, img_cols, rows, cols):
  counter = np.zeros((rows*cols))
  roi_rows = img_rows // rows
  roi_cols = img_cols // cols
  n = len(no_match_points)
  for i in range(n):
    col, row = no_match_points[i]
    col = min(img_cols-1, max(0, col))
    row = min(img_rows-1, max(0, row))
    index = cols * (int(row) // roi_rows) + (int(col) // roi_cols)
    counter[index] += 1
  return counter

def project_points_with_homography(points_old, H):
  n = points_old.shape[0]
  points_current = np.zeros(points_old.shape)
  points_old_hc = cv2.convertPointsToHomogeneous(np.float32(points_old))
  points_old_hc = np.reshape(points_old_hc, (n, 3))

  for i in range(n):
    point_old = np.transpose(points_old_hc[i])
    point_current_hc = np.dot(H, point_old)
    points_current[i, 0] = point_current_hc[0] / (point_current_hc[2]+1e-6)
    points_current[i, 1] = point_current_hc[1] / (point_current_hc[2]+1e-6)

  return points_current

def compute_correlation(points_old, img_old, img_current, patch_size, H):
  rows, cols = img_current.shape
  n = len(points_old)
  correlation = np.zeros(n, dtype=np.float32)
  patch_size_half = int(patch_size/2)

  for k in range(n):
    row = int(points_old[k][1])
    col = int(points_old[k][0])
    patch_old = np.zeros((patch_size, patch_size), dtype = np.float32)
    
    if row < patch_size_half or row > rows-1-patch_size_half:
      correlation[k] = 0.0
      continue
    elif col < patch_size_half or col > cols-1-patch_size_half:
      correlation[k] = 0.0
      continue
    else:
      patch_old[...] = img_old[row-patch_size_half:row+patch_size_half+1, col-patch_size_half:col+patch_size_half+1]

    # Proyectar patch
    patch_curr = np.zeros((patch_size, patch_size), dtype = np.float32)
    point_old = np.zeros((1, 2))

    for i in range(patch_size):
      for j in range(patch_size):
        point_old[0, 0] = col-patch_size_half+i
        point_old[0, 1] = row-patch_size_half+j
        point_curr = project_points_with_homography(point_old, H)
        index_row = round(point_curr[0, 1])
        index_col = round(point_curr[0, 0])
        if index_row >= rows or index_col >= cols or index_row < 0 or index_col < 0:
          patch_curr[j, i] = patch_old[j, i]
        else:
          patch_curr[j, i] = img_current[index_row, index_col]
    
    #Resultado
    correlation[k] = np.sum(np.abs(patch_curr - patch_old)) / (patch_size * patch_size)

  return correlation