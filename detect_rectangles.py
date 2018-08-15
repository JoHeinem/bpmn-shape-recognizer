# USAGE
# python detect_circles.py --image images/simple.png
import cv2
import imutils
import pkg_resources
import numpy as np

# import the necessary packages

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
# image_filename = pkg_resources.resource_filename('resources', 'shapes_and_colors.png')
# image_filename = pkg_resources.resource_filename('resources', 'test.png')


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename, )
ratio = image.shape[0] / 800.0
orig = image.copy()
image = imutils.resize(image, height = 800)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
screenCnt = None

# loop over our contours
for c in cnts:
  # approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, True)

  # if our approximated contour has four points, then
  # we can assume that we have found our screen
  if len(approx) == 4:
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)


cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)
