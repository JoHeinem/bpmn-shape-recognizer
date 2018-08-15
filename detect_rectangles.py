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
image_filename = pkg_resources.resource_filename('resources', 'foo.png')


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename, )
orig = image.copy()
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# cv2.imshow("Canny", edged)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
screenCnt = None

# loop over our contours
for c in cnts:
  # approximate the contour
  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.04 * peri, True)

  # if our approximated contour has four points, then
  # we can assume that we have found our screen
  (x, y, w, h) = cv2.boundingRect(approx)
  ar = w / float(h)
  rect_area = w * float(h)
  area = cv2.contourArea(c)
  # shape = "arrow" if area < rect_area / 2 else shape

  if len(approx) == 4 and area > rect_area / 3:

    # assumption the x coordinates of the first and third point
    # and the y coordinates for the second and forth point are the same
    # for a diamond shape.
    diamont_distance = abs(approx[0][0][0] - approx[2][0][0]) + abs(approx[1][0][1] - approx[3][0][1])
    rect_distance = abs(approx[0][0][0] - approx[1][0][0]) + abs(approx[2][0][0] - approx[3][0][0])
    if diamont_distance < rect_distance:
        shape = "diamond"
    else:
        shape = 'rectangle'

    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    cx = int(x + w/2)
    cy = int(y + h/2)
    cv2.rectangle(image, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)


cv2.imshow("Game Boy Screen", image)
cv2.waitKey(0)
