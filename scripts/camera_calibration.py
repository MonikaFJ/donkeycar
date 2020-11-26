import numpy as np
import cv2
import glob
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
filenames = glob.glob(dir_path + '/../data/fisheye_calibration/*.jpg')
images = []
i = 0

# Checkboard dimensions
CHECKERBOARD = (6,9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
found_count = 0
### read images and for each image:
for fname in filenames:
    print("image: " + fname)
    img = cv2.imread(fname)
    img_shape = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("points found " + fname)
        found_count = found_count +1
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(corners)
###

# calculate K & D
N_imm = found_count
K = np.zeros((3, 3))
D = np.zeros((4, 1))

rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

print(K)
print(D)

mat = np.matrix(K)
with open(dir_path + '/../data/fisheye_calibration/K.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.10f')

mat = np.matrix(D)
with open(dir_path + '/../data/fisheye_calibration/D.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.10f')

mat = np.matrix(img_shape)
with open(dir_path + '/../data/fisheye_calibration/DIM.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%i')
