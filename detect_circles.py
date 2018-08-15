# USAGE
# python detect_circles.py --image images/simple.png
import imutils
import pkg_resources

import cv2
# import the necessary packages
import numpy as np

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
# image_filename = pkg_resources.resource_filename('resources', 'simple_draw.png')
# image_filename = pkg_resources.resource_filename('resources', 'test.png')


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.8, minDist=200, param1=30, param2=75, minRadius=5, maxRadius=70)

# ensure at least some circles were found
if circles is not None:
  # convert the (x, y) coordinates and radius of the circles to integers
  circles = np.round(circles[0, :]).astype("int")

  # loop over the (x, y) coordinates and radius of the circles
  for (x, y, r) in circles:
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

  # show the output image
  cv2.imshow("output", np.hstack([image, output]))
  cv2.waitKey(0)
