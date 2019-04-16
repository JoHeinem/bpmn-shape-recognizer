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
image_filename = pkg_resources.resource_filename('resources', 'foo2.png')
# image_filename = pkg_resources.resource_filename('resources', 'open_rectangle.png')
# image_filename = pkg_resources.resource_filename('resources', 'original.png')



# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,5,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()