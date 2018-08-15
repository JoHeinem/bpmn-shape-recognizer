# USAGE
# python detect_circles.py --image images/simple.png
import imutils
import pkg_resources

import cv2
# import the necessary packages
import numpy as np
try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
# image_filename = pkg_resources.resource_filename('resources', 'simple_draw.png')
# image_filename = pkg_resources.resource_filename('resources', 'test.png')


print(pytesseract.image_to_string(Image.open(image_filename), lang='eng'))

# TODO: find point furthest away from center -> p1
# 2. find point furthest away from p1 -> p2
# 3. find point furstest away from p1 & p2 -> p3
# 4. find point furthest away from p1, p2 & p3 -> p4
# connect p1 and p2 -> 50% of line should be in x
# connect p3 and p4 -> 50 % of line should be in x
