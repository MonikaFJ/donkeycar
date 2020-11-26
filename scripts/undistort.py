import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt


class Undistorter:
  def __init__(self, img_dim):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    self.img_dim = img_dim
    K = np.loadtxt(dir_path + '/../data/fisheye_calibration/K.txt', delimiter=" ")
    D = np.loadtxt(dir_path + '/../data/fisheye_calibration/D.txt', delimiter=" ")
    DIM = np.loadtxt(dir_path + '/../data/fisheye_calibration/DIM.txt', delimiter=" ")
    # print(K)
    # print(D)
    # print(DIM)

    scaled_K = K * self.img_dim[0] / DIM[0]
    scaled_K[2][2] = 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
                                                                   img_dim, np.eye(3), balance=0.5)
    self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
                                                               new_K, img_dim, cv2.CV_16SC2)

  def undistort(self, img):
    return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)

  def warp_image(self, img):
    height = self.img_dim[0]
    width = self.img_dim[1]

    src = np.float32(
      [[(width / 2) - 55, height / 2 + 100],
       [((width / 6) - 10), height],
       [(width * 5 / 6) + 60, height],
       [(width / 2 + 55), height / 2 + 100]])
    dst = np.float32(
      [[(width / 4), 0],
       [(width / 4), height],
       [(width * 3 / 4), height],
       [(width * 3 / 4), 0]])

    src = np.float32(
      [[(width / 2) - 55, height / 2 + 100],
       [((width / 6) - 10), height],
       [(width * 5 / 6) + 60, height],
       [(width / 2 + 55), height / 2 + 100]])
    dst = np.float32(
      [[(width / 4), 0],
       [(width / 4), height],
       [(width * 3 / 4), height],
       [(width * 3 / 4), 0]])


    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)

    return warped, M

  def four_point_transform(self, image):
    # obtain a consistent order of the points and unpack them
    # individually
    point_x = 0
    point = 500
    y_point = 400
    y_point_max = 900
    tl = [point_x, y_point]
    tr = [self.img_dim[0] - point_x, y_point]
    br = [self.img_dim[0]-point, y_point_max]
    bl = [point, y_point_max]
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = tl
    rect[1] = tr
    rect[2] = br
    rect[3] = bl
    #rect = np.array([tl, tr, br, bl])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

#

dir_path = os.path.dirname(os.path.realpath(__file__))

img = cv2.imread(dir_path + '/../data/line/test_line_1.jpg')
img_dim = img.shape[:2][::-1]
undistort = Undistorter(img_dim)
undist_image = undistort.undistort(img)
imgplot = plt.imshow(undist_image)
plt.show()

binary_warped = undistort.four_point_transform(undist_image)
imgplot = plt.imshow(binary_warped)
plt.show()