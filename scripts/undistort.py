import numpy as np
import cv2
import glob

K = np.loadtxt(dir_path + '/data/fisheye_calibration/K.txt', delimiter=" ")
D = np.loadtxt(dir_path + '/data/fisheye_calibration/D.txt', delimiter=" ")
DIM = np.loadtxt(dir_path + '/data/fisheye_calibration/DIM.txt', delimiter=" ")
print(K)
print(D)
print(DIM)

img = cv2.imread('/home/florek/Desktop/donkey/donkeycar/fisheye_calibration/1.jpg')
img_dim = img.shape[:2][::-1]

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (1944, 2592), cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
undistorted_img = cv2.resize(undistorted_img, (img_dim[0]//2, img_dim[1]//2))
cv2.imshow('img',undistorted_img)
cv2.waitKey(5000)


balance = 0.5


scaled_K = K * img_dim[0] / DIM[0]
scaled_K[2][2] = 1.0
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
    img_dim, np.eye(3), balance=balance)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
    new_K, img_dim, cv2.CV_16SC2)
undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT)
undist_image = cv2.resize(undist_image, (img_dim[0]//2, img_dim[1]//2))

cv2.imshow('img',undist_image)
cv2.waitKey(50000)
