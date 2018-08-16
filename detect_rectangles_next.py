# USAGE
# python detect_circles.py --image images/simple.png
import imutils
import pkg_resources

import cv2
# import the necessary packages
import numpy as np
import math

# construct the argument parse and parse the arguments
image_filename = pkg_resources.resource_filename('resources', 'hand_drawn_bpmn_shapes.png')
# image_filename = pkg_resources.resource_filename('resources', 'simple_draw.png')
# image_filename = pkg_resources.resource_filename('resources', 'test.png')
# image_filename = pkg_resources.resource_filename('resources', 'foo.png')
# image_filename = pkg_resources.resource_filename('resources', 'open_rectangle.png')

def calc_dist(p2, p1):
      return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)
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
        if calc_dist(p2, p1) < 10:
            should_add = False

    if should_add:
        result = np.vstack((result, p1))

for ith_corner in range(result.shape[0]):
    corner = result[ith_corner]
    sum =0
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






# find most left upper corner
# try to find the remaining corner
# if found remove all 4 corners from list
# if not found remove current corner point
# rectanlges =
# MAX_DIST = 20
# while(np.size(result) != 0):
#     upper_left_corner_idx = np.argmin(np.sum(result, axis=1))
#     upper_left_corner_point = result[upper_left_corner_idx]
#     possible_lower_left_corner_idx = []
#     for lower_left_ith in [x for x in range(result.shape[0]) if x != upper_left_corner_idx]:
#         dist = abs(result[lower_left_ith][0] - upper_left_corner_point[0])
#         if dist < MAX_DIST:
#             possible_lower_left_corner_idx.append(lower_left_ith)
#
#     idx_to_remove = [upper_left_corner_idx]
#     possible_lower_right_corners_idx = []
#     for ith_lower_left in possible_lower_left_corner_idx:
#         for lower_right_ith in [x for x in range(result.shape[0]) if x != upper_left_corner_idx and x not in possible_lower_left_corner_idx]:
#             dist = abs(result[lower_right_ith][1] - result[ith_lower_left][1])
#             if dist < MAX_DIST:
#                 # find remaining upper right corner
#                 for upper_right_ith in [x for x in range(result.shape[0]) if x != upper_left_corner_idx and x not in possible_lower_left_corner_idx and x != lower_right_ith]:
#                     dist_x = abs(result[lower_right_ith][0] - result[upper_right_ith][0])
#                     dist_y = abs(result[upper_left_corner_idx][1] - result[upper_right_ith][1])
#                     if dist_x < MAX_DIST and dist_y < MAX_DIST:
#                         x = result[upper_left_corner_idx][0]
#                         y = result[upper_left_corner_idx][1]
#                         w = result[upper_right_ith][0] - x
#                         h = result[ith_lower_left][1] - y
#                         cv2.rectangle(image, (x , y), (x+w, y+h), (0, 255, 0), 3)
#                         idx_to_remove.append(ith_lower_left)
#                         idx_to_remove.append(lower_right_ith)
#                         idx_to_remove.append(upper_right_ith)
#
#     to_choose = [x for x in range(result.shape[0]) if x not in idx_to_remove]
#     result = result[to_choose, :]


# result


# cv2.imshow('foo', dst)

# Threshold for an optimal value, it may vary depending on the image.
image[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()