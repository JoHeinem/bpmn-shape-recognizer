# import the necessary packages
import math

import cv2
import imutils
import numpy as np
from uuid import uuid4
from backend.pyimagesearch.id_generator import IdGenerator

class ShapeDetector:
  def __init__(self):
    pass

  def detect_all_shapes(self, image):
    all_shapes = {}
    circles = self.detect_circles(image)
    rectangles = self.detect_rectangles(image)
    all_shapes.update(circles)
    all_shapes.update(rectangles)
    seq_flows = self.detect_seqflows(image, all_shapes)
    all_shapes.update(seq_flows)
    return all_shapes

  def detect_circles(self, image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.8, minDist=200, param1=30, param2=75, minRadius=5, maxRadius=70)

    # ensure at least some circles were found
    result_circles = {}
    if circles is not None:
      # convert the (x, y) coordinates and radius of the circles to integers
      circles = np.round(circles[0, :]).astype("int")

      # loop over the (x, y) coordinates and radius of the circles
      for (x, y, r) in circles:
        # draw the circirclescle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        id = 'event_' + IdGenerator.next()
        r = r.item() # convert to python
        x = x.item() - r
        y = y.item() - r
        result_circles[id] = {'type': 'bpmn:StartEvent', 'x': x, 'y': y, 'width': 2*r, 'height': 2*r}


    return result_circles

  def connect_gaps(self, image):

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
            if self.calc_dist(p2, p1) < 15:
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

        for ith_point in [x for x in range(result.shape[0]) if x != ith_corner]:
            point = result[ith_point]
            dist = self.calc_dist(corner, point)
            if dist < 20:
                lineThickness = 2
                cv2.line(image, (corner[1], corner[0]), (point[1], point[0]), (0,0,0), lineThickness)
                break
    return image


  def detect_rectangles(self, image):
    image = self.connect_gaps(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # cv2.imshow('foo', edged)
    # cv2.waitKey(0)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour

    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    screenCnt = None

    rectangles = {}
    center_points = []
    # loop over our contours
    for c in cnts:
      # approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)

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
        # x = approx[0][0][0]
        # y = approx[0][0][1]

        cx = int(x + w/2)
        cy = int(y + h/2)
        new_center_point = [cx, cy]
        should_add = True
        for center_point in center_points:
          dist = self.calc_dist(new_center_point, center_point)
          if dist < 15:
            should_add = False

        if should_add:
          center_points.append([cx, cy])
          id = 'task_' + IdGenerator.next()
          x = x.item()
          y = y.item()
          rectangles[id] = {'type': 'bpmn:Task', 'x': x, 'y': y, 'width': w, 'height': h}
    return rectangles

  def detect_seqflows(self, original_image, elems):
      # assumption: all other forms have been detected first and are
      # contained in the elems

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
      seq_flows = {}
      for c in cnts:
        shape = self.detect(c)
        if shape == 'arrow':

          peri = cv2.arcLength(c, True)
          approx = cv2.approxPolyDP(c, 0.04 * peri, True)

          from_id = self.find_closest(approx[0][0], elems)
          to_id = self.find_closest(approx[len(approx)-1][0], elems)
          waypoints = []
          for ith in range(len(approx)):
            ith_elem = approx[ith][0]
            waypoint = {
              'x': ith_elem[0].item(),
              'y': ith_elem[1].item()
            }
            waypoints.append(waypoint)

          id = 'seqflow_' + IdGenerator.next()
          seq_flows[id] = {'type': 'bpmn:SequenceFlow',
                           'waypoints': waypoints,
                            'connects': {'from': from_id, 'to': to_id}
                           }
      return seq_flows

  def find_closest(self, point, elems):
    closest_id = ''
    dist = 1000000000000
    for key, elem in elems.items():
      cx = elem['x'] + elem['width']/2
      cy = elem['y'] + elem['height']/2
      new_dist = self.calc_dist(point, [cx, cy])
      if new_dist<dist:
        closest_id = key
        dist = new_dist

    return None if closest_id == '' else closest_id




  def detect(self, c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices

    if len(approx) == 3:
      p1 = approx[0][0]
      p2 = approx[1][0]
      p3 = approx[2][0]

      dists = [self.calc_dist(p1, p2), self.calc_dist(p2, p3), self.calc_dist(p3, p1)]
      max_dist = max(dists)
      min_dist = min(dists)
      shape = "arrow" if min_dist < max_dist / 2 else "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
      # compute the bounding box of the contour and use the
      # bounding box to compute the aspect ratio
      (x, y, w, h) = cv2.boundingRect(approx)

      # assumption the x coordinates of the first and third point
      # and the y coordinates for the second and forth point are the same
      # for a diamond shape.
      diamont_distance = abs(approx[0][0][0] - approx[2][0][0]) + abs(approx[1][0][1] - approx[3][0][1])
      rect_distance = abs(approx[0][0][0] - approx[1][0][0]) + abs(approx[2][0][0] - approx[3][0][0])

      if diamont_distance < rect_distance:
        shape = "diamond"
      else:
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        rect_area = w * float(h)
        area = cv2.contourArea(c)
        shape = "arrow" if area < rect_area / 2 else shape

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
      (x, y, w, h) = cv2.boundingRect(approx)
      approx_penta_area = w * float(h)
      area = cv2.contourArea(c)
      shape = "arrow" if area < approx_penta_area / 2 else "pentagon"

    # otherwise, we assume the shape is a circle
    else:

      shape = 'arrow' if self.is_arrow(c) else 'circle'

    # return the name of the shape
    return shape

  def calc_dist(self, p2, p1):
      return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

  def is_arrow(self, contour):
    def find_furthest(p, all_points):
      furth_p = p
      for ith in all_points:
        ith = ith[0]
        curr_furth_dist = abs(furth_p[0] - p[0]) + abs(furth_p[1] - p[1])
        ith_dist = abs(ith[0] - p[0]) + abs(ith[1] - p[1])
        furth_p = ith if ith_dist > curr_furth_dist else furth_p
      return furth_p



    # find approximately furthest points from each other
    # find average
    avg_x = 0
    avg_y = 0
    for point in contour:
      point = point[0]
      avg_x += point[0]
      avg_y += point[1]
    avg_x /= len(contour)
    avg_y /= len(contour)

    # find furthest point from average
    furthest = find_furthest([avg_x, avg_y], contour)
    furthest_from_furthest = find_furthest(furthest, contour)

    # calculate sum of distance all points to line between two furthest points
    # sum_dist = 0
    # for point in approx:
    #   point = point[0]
    #   nominator = abs(
    #     (furthest_from_furthest[1] - furthest[1]) * point[0] -
    #     (furthest_from_furthest[0] - furthest[0]) * point[1] +
    #     furthest_from_furthest[0] * furthest[1] -
    #     furthest_from_furthest[1] * furthest[0]
    #   )
    #   denominator = calc_dist(furthest_from_furthest, furthest)
    #   distance = nominator/denominator
    #   sum_dist += distance

    area = cv2.contourArea(contour)
    furthest_dist = self.calc_dist(furthest_from_furthest, furthest)
    circle_area = math.pi * (furthest_dist / 2) ** 2
    return area < circle_area / 2
