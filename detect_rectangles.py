# USAGE
# python detect_circles.py --image images/simple.png
import cv2
import imutils
import pkg_resources
import numpy as np
import math

# import the necessary packages

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
image_filename = pkg_resources.resource_filename('resources', 'shapes_and_colors.png')
# image_filename = pkg_resources.resource_filename('resources', 'open_rectangle.png')
# image_filename = pkg_resources.resource_filename('resources', 'foo.png')
image_filename = pkg_resources.resource_filename('resources', 'original.png')

def calc_dist(p2, p1):
      return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

def connect_gaps(image):

  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

  gray = np.float32(gray)
  dst = cv2.cornerHarris(gray,2,5,0.04)

  #result is dilated for marking the corners, not important
  dst = cv2.dilate(dst,None)

  corner_indexes = np.where(dst>0.01*dst.max())
  result = np.array([[corner_indexes[0][0], corner_indexes[1][0]]])
  for lower_right_ith in range(len(corner_indexes[0])):
      p1 = np.array([corner_indexes[0][lower_right_ith], corner_indexes[1][lower_right_ith]])
      should_add = True
      for p2 in result:
          if calc_dist(p2, p1) < 15:
              should_add = False

      if should_add:
          result = np.vstack((result, p1))

  for ith_corner in range(result.shape[0]):
      corner = result[ith_corner]
      sum =0
      # TODO: check that is not at corner of image
      sum += gray[corner[0]-1, corner[1]-1]
      sum += gray[corner[0], corner[1]-1]
      sum += gray[corner[0]+1, corner[1]-1]
      sum += gray[corner[0]-1, corner[1]]
      # sum += gray[corner[0], corner[1]]
      sum += gray[corner[0]+1, corner[1]]
      sum += gray[corner[0]-1, corner[1]+1]
      sum += gray[corner[0], corner[1]+1]
      sum += gray[corner[0]+1, corner[1]+1]

      closest = np.array([100000000000, 10000000000])
      for ith_point in [x for x in range(result.shape[0]) if x != ith_corner]:
          point = result[ith_point]
          dist = calc_dist(corner, point)
          if dist < 20:
              lineThickness = 2
              cv2.line(image, (corner[1], corner[0]), (point[1], point[0]), (0,0,0), lineThickness)
              break
  return image


# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename, )
image = connect_gaps(image)
orig = image.copy()
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

cv2.imshow("Canny", edged)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour

cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
screenCnt = None

center_points = []
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

  if len(approx) == 4:

    # assumption the x coordinates of the first and third point
    # and the y coordinates for the second and forth point are the same
    # for a diamond shape.
    diamont_distance = abs(approx[0][0][0] - approx[2][0][0]) + abs(approx[1][0][1] - approx[3][0][1])
    rect_distance = abs(approx[0][0][0] - approx[1][0][0]) + abs(approx[2][0][0] - approx[3][0][0])
    if diamont_distance < rect_distance:
        shape = "diamond"
    else:
        shape = 'rectangle'

    cx = int(x + w/2)
    cy = int(y + h/2)
    new_center_point = [cx, cy]

    should_add = True
    for center_point in center_points:
      dist = calc_dist(new_center_point, center_point)
      if dist < 15:
        should_add = False

    if should_add:
      center_points.append([cx, cy])
      # cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
      cv2.rectangle(image, (x , y), (x+w, y+h), (0, 255, 0), 3)
      cv2.rectangle(image, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

print(center_points)
cv2.imshow("image", image)
cv2.waitKey(0)
