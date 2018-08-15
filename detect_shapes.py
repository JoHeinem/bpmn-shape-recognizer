# USAGE
# python detect_shapes.py --image shapes_and_colors.png

import cv2
import imutils
import numpy as np
import pkg_resources

# import the necessary packages
from backend.pyimagesearch.shapedetector_backup import ShapeDetector

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
output_filename = pkg_resources.resource_filename('resources', 'output.png')


# image_filename = pkg_resources.resource_filename('resources', 'shapes_and_colors.png')
# image_filename = pkg_resources.resource_filename('resources', 'test.png')
image_filename = pkg_resources.resource_filename('resources', 'foo.png')

# To execute the script, do:




def detect_shapes(original_image):
  # image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
  gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
  _, image = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

  # Copy the thresholded image.
  im_floodfill = image.copy()

  h, w = image.shape[:2]
  mask = np.zeros((h + 2, w + 2), np.uint8)

  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0, 0), 255)

  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # Combine the two images to get the foreground.
  image = image | im_floodfill_inv

  # resized = imutils.resize(image, width=800)
  # ratio = image.shape[0] / float(resized.shape[0])
  ratio = 1

  # convert the resized image to grayscale, blur it slightly,
  # and threshold it
  # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(image, (5, 5), 0)
  thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

  # find contours in the thresholded image and initialize the
  # shape detector
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]
  sd = ShapeDetector()

  # loop over the contours
  for c in cnts:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(original_image, [c], -1, (0, 255, 0), 2)
    cv2.putText(original_image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 0, 200), 2)

    # show the output image
    # cv2.imshow("Image", original_image)

  cv2.imwrite(output_filename, original_image)
  cv2.imshow("image", original_image)
  cv2.waitKey(0)

# original_image = cv2.imread(image_filename, )
# detect_shapes(original_image)