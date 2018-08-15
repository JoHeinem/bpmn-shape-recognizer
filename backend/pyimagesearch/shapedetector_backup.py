# import the necessary packages
import math

import cv2
import numpy as np


class ShapeDetector:
  def __init__(self):
    pass

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
