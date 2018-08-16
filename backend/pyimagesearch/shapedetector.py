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
            if self.calc_dist(p2, p1) < 3:
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
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,5,0.04)

    # cv2.imshow("Gray", gray)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    corner_indexes = np.where(dst>0.01*dst.max())
    result = np.array([[corner_indexes[0][0], corner_indexes[1][0]]])
    for ith_corner in range(len(corner_indexes[0])):
        p1 = np.array([corner_indexes[0][ith_corner], corner_indexes[1][ith_corner]])
        should_add = True
        if gray[p1[0], p1[1]] == 255:
            should_add = False
        else:
            for p2 in result:
                if self.calc_dist(p2, p1) < 30:
                    should_add = False

        if should_add:
            result = np.vstack((result, p1))

    result_copy = []
    for r in result:
        result_copy.append(r)

    center_points = []
    rectangles = {}
    while(np.size(result) != 0):
        upper_left_corner_idx = np.argmin(np.sum(result, axis=1))
        upper_left_corner_point = result[upper_left_corner_idx]

        # the corner point is not directly on the edge
        if gray[upper_left_corner_point[0]+1, upper_left_corner_point[1]] != 0:
            upper_left_corner_point = [upper_left_corner_point[0], upper_left_corner_point[1]+1]

        idx_to_remove = [upper_left_corner_idx]

        can_continue = True
        next_point = upper_left_corner_point
        max_right = 10
        while can_continue:
            left_down = gray[next_point[0]+1, next_point[1]-1]
            down = gray[next_point[0]+1, next_point[1]]
            right_down = gray[next_point[0]+1, next_point[1]+1]
            right = gray[next_point[0], next_point[1]+1]
            if down == 0:
                next_point = [next_point[0]+1, next_point[1]]
            elif right_down == 0:
                next_point = [next_point[0]+1, next_point[1]+1]
            elif left_down == 0:
                next_point = [next_point[0]+1, next_point[1]-1]
            elif right == 0 and max_right>0:
                max_right -= 1
                next_point = [next_point[0], next_point[1]+1]
            else:
              can_continue = False

        # TODO: calc dist
        can_continue = True
        lower_left_corner = next_point
        max_up = 10
        while can_continue:
            right_up = gray[next_point[0]-1, next_point[1]+1]
            right = gray[next_point[0], next_point[1]+1]
            right_down = gray[next_point[0]+1, next_point[1]+1]
            up = gray[next_point[0]-1, next_point[1]]
            if right == 0:
                next_point = [next_point[0], next_point[1]+1]
            elif right_up == 0:
                next_point = [next_point[0]-1, next_point[1]+1]
            elif right_down == 0:
                next_point = [next_point[0]+1, next_point[1]+1]
            elif up == 0 and max_up > 0:
                max_up -= 1
                next_point = [next_point[0]-1, next_point[1]]
            else:
              can_continue = False
        # TODO: calc dist
        can_continue = True
        lower_right_corner = next_point
        max_left = 10
        while can_continue:
            left_up = gray[next_point[0]-1, next_point[1]-1]
            up = gray[next_point[0]-1, next_point[1]]
            right_up = gray[next_point[0]-1, next_point[1]+1]
            left = gray[next_point[0], next_point[1]-1]
            if up == 0:
                next_point = [next_point[0]-1, next_point[1]]
            elif left_up == 0:
                next_point = [next_point[0]-1, next_point[1]-1]
            elif right_up == 0:
                next_point = [next_point[0]-1, next_point[1]+1]
            elif left == 0 and max_left>0:
                max_left -= 1
                next_point = [next_point[0], next_point[1]-1]
            else:
              can_continue = False

        can_continue = True
        upper_right_corner = next_point
        while can_continue:
            left_up = gray[next_point[0]-1, next_point[1]-1]
            left = gray[next_point[0], next_point[1]-1]
            left_down = gray[next_point[0]+1, next_point[1]-1]
            if left == 0:
                next_point = [next_point[0], next_point[1]-1]
            elif left_down == 0:
                next_point = [next_point[0]+1, next_point[1]-1]
            elif left_up == 0:
                next_point = [next_point[0]-1, next_point[1]-1]
            else:
              can_continue = False

        dist = self.calc_dist(upper_left_corner_point, next_point)
        dist_left = self.calc_dist(upper_left_corner_point, lower_left_corner)
        dist_bottom = self.calc_dist(lower_right_corner, lower_left_corner)
        dist_right = self.calc_dist(lower_right_corner, upper_right_corner)
        if dist < 40 and dist_left > 40 and dist_bottom > 40 and dist_right > 40:
            x1 = upper_left_corner_point[0]
            x2 = upper_right_corner[0]
            y1 = upper_left_corner_point[1]
            y2 = lower_left_corner[1]
            x = int((x1+x2)/2)
            y = int((y1+y2)/2)
            w = int((upper_right_corner[1]+lower_right_corner)[1]/2) - y
            h = int((lower_left_corner[0]+lower_right_corner[0]) /2)- x
            cv2.rectangle(image, (y , x), (y+w, x+h ), (0, 255, 0), 3)

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
              x = x
              y = y
              rectangles[id] = {'type': 'bpmn:Task', 'x': y, 'y': x, 'width': w, 'height': h}

        to_choose = [x for x in range(result.shape[0]) if x not in idx_to_remove]
        result = result[to_choose, :]

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

          # find most left
          most_left_idx = 0
          for ith in range(len(approx)):
            curr_x = approx[ith][0][0]
            most_x = approx[most_left_idx][0][0]
            if curr_x < most_x:
              most_left_idx = ith

          most_right_idx = 0
          for ith in range(len(approx)):
            curr_x = approx[ith][0][0]
            most_x = approx[most_right_idx][0][0]
            if curr_x > most_x:
              most_right_idx = ith

          from_id = self.find_closest(approx[most_left_idx][0], elems)
          to_id = self.find_closest(approx[most_right_idx][0], elems)
          waypoints = []
          for ith in range(len(approx)):
            ith_elem = approx[ith][0]
            waypoint = {
              'x': ith_elem[0].item(),
              'y': ith_elem[1].item()
            }
            waypoints.append(waypoint)
          waypoints = sorted(waypoints, key=lambda k: k['x'])

          seq_dist = self.calc_dist([waypoints[0]['x'], waypoints[0]['y']], [waypoints[len(waypoints)-1]['x'], waypoints[len(waypoints)-1]['y']])
          if seq_dist < 50:
            break

          # remove waypoints close to target
          to_remove = []
          last_waypoint = [waypoints[len(waypoints)-1]['x'], waypoints[len(waypoints)-1]['y']]
          for ith in range(1, len(waypoints)-1):
            curr_waypoint_x = waypoints[ith]['x']
            curr_waypoint_y = waypoints[ith]['y']
            dist = self.calc_dist([curr_waypoint_x, curr_waypoint_y], last_waypoint)
            if dist<40:
              to_remove.append(ith)

          to_choose = [x for x in range(len(waypoints)) if x not in to_remove]
          waypoints = [waypoints[i] for i in to_choose]

          if from_id is not None:
            from_elem = elems[from_id]
            waypoints[0]['x'] = from_elem['x'] + from_elem['width']
            waypoints[0]['y'] = from_elem['y'] + from_elem['height']/2

          if to_id is not None:
            to_elem = elems[to_id]
            waypoints[len(waypoints)-1]['x'] = to_elem['x']
            waypoints[len(waypoints)-1]['y'] = to_elem['y'] + to_elem['height']/2

          id = 'seqflow_' + IdGenerator.next()
          seq_flows[id] = {'type': 'bpmn:SequenceFlow',
                           'waypoints': waypoints,
                            'connects': {'from': from_id, 'to': to_id}
                           }
      return seq_flows

  def find_closest(self, point, elems):
    closest_id = ''
    dist = 100
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
