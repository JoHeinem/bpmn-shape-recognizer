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
# image_filename = pkg_resources.resource_filename('resources', 'foo2.png')
# image_filename = pkg_resources.resource_filename('resources', 'foo3.png')
# image_filename = pkg_resources.resource_filename('resources', 'foo4.png')
# image_filename = pkg_resources.resource_filename('resources', 'open_rectangle.png')
image_filename = pkg_resources.resource_filename('resources', 'original.png')

def calc_dist(p2, p1):
      return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)

# load the image, clone it for output, and then convert it to grayscale
image = cv2.imread(image_filename)
# ratio = image.shape[0] / 800.0
# image = imutils.resize(image, height = 800)
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
    # TODO: 
    if gray[p1[0], p1[1]] == 255:
        should_add = False
    else:
        for p2 in result:
            if calc_dist(p2, p1) < 30:
                should_add = False

    if should_add:
        result = np.vstack((result, p1))

# for ith_corner in range(result.shape[0]):
#     corner = result[ith_corner]
#     sum =0
#     sum += gray[corner[0]-1, corner[1]-1]
#     sum += gray[corner[0], corner[1]-1]
#     sum += gray[corner[0]+1, corner[1]-1]
#     sum += gray[corner[0]-1, corner[1]]
#     # sum += gray[corner[0], corner[1]]
#     sum += gray[corner[0]+1, corner[1]]
#     sum += gray[corner[0]-1, corner[1]+1]
#     sum += gray[corner[0], corner[1]+1]
#     sum += gray[corner[0]+1, corner[1]+1]
#
#     closest = np.array([100000000000, 10000000000])
#     for ith_point in [x for x in range(result.shape[0]) if x != ith_corner]:
#         point = result[ith_point]
#         dist = calc_dist(corner, point)
#         if dist < 20:
#             lineThickness = 2
#             cv2.line(image, (corner[1], corner[0]), (point[1], point[0]), (0,0,0), lineThickness)
#             break

diamond_corners = []
for r in result:
    diamond_corners.append(r)
diamond_corners = np.array(diamond_corners)

coners_copy = []
for r in result:
    coners_copy.append(r)
coners_copy = np.array(coners_copy)

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

    dist = calc_dist(upper_left_corner_point, next_point)
    dist_left = calc_dist(upper_left_corner_point, lower_left_corner)
    dist_bottom = calc_dist(lower_right_corner, lower_left_corner)
    dist_right = calc_dist(lower_right_corner, upper_right_corner)
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
    to_choose = [x for x in range(result.shape[0]) if x not in idx_to_remove]
    result = result[to_choose, :]

while(np.size(diamond_corners) != 0):
    top_idx = np.argmin(diamond_corners, axis=0)[0]
    top_corner = diamond_corners[top_idx]

    # the corner point is not directly on the edge
    if gray[top_corner[0]+1, top_corner[1]] != 0:
        top_corner = [top_corner[0]+3, top_corner[1]]

    idx_to_remove = [top_idx]

    can_continue = True
    next_point = top_corner
    max_left = 20
    while can_continue:
        left_down = gray[next_point[0]+1, next_point[1]-1]
        down = gray[next_point[0]+1, next_point[1]]
        left = gray[next_point[0], next_point[1]-1]
        if left_down == 0:
            next_point = [next_point[0]+1, next_point[1]-1]
        elif down == 0:
            next_point = [next_point[0]+1, next_point[1]]
        elif left == 0 and max_left>0:
            max_left -= 1
            next_point = [next_point[0], next_point[1]-1]
        else:
          can_continue = False

    # TODO: calc dist
    can_continue = True
    left_corner = next_point
    max_down = 20
    max_up = 10
    while can_continue:
        down = gray[next_point[0]+1, next_point[1]]
        right = gray[next_point[0], next_point[1]+1]
        right_down = gray[next_point[0]+1, next_point[1]+1]
        right_up = gray[next_point[0]-1, next_point[1]+1]
        if right_down == 0:
            next_point = [next_point[0]+1, next_point[1]+1]
        elif right == 0:
            next_point = [next_point[0], next_point[1]+1]
        elif down == 0 and max_down>0:
            max_down -= 1
            next_point = [next_point[0]+1, next_point[1]]
        elif right_up == 0 and max_up>0:
            max_up -= 1
            next_point = [next_point[0]-1, next_point[1]+1]
        else:
          can_continue = False
    # TODO: calc dist
    can_continue = True
    bottom_corner = next_point
    max_right = 20
    while can_continue:
        right = gray[next_point[0], next_point[1]+1]
        up = gray[next_point[0]-1, next_point[1]]
        right_up = gray[next_point[0]-1, next_point[1]+1]
        if right_up == 0:
            next_point = [next_point[0]-1, next_point[1]+1]
        elif up == 0:
            next_point = [next_point[0]-1, next_point[1]]
        elif right == 0 and max_right>0:
            max_right -= 1
            next_point = [next_point[0], next_point[1]+1]
        else:
          can_continue = False

    can_continue = True
    right_corner = next_point
    max_up = 20
    max_down = 10
    while can_continue:
        left_up = gray[next_point[0]-1, next_point[1]-1]
        left = gray[next_point[0], next_point[1]-1]
        up = gray[next_point[0]-1, next_point[1]]
        left_down = gray[next_point[0]+1, next_point[1]-1]
        if left_up == 0:
            next_point = [next_point[0]-1, next_point[1]-1]
        elif left == 0:
            next_point = [next_point[0], next_point[1]-1]
        elif up == 0 and max_up >0:
            max_up -= 1
            next_point = [next_point[0]-1, next_point[1]]
        elif left_down == 0 and max_down>0:
            max_down -= 1
            next_point = [next_point[0]+1, next_point[1]-1]
        else:
          can_continue = False

    dist = calc_dist(top_corner, next_point)
    dist_left = calc_dist(top_corner, left_corner)
    dist_bottom = calc_dist(bottom_corner, left_corner)
    dist_right = calc_dist(bottom_corner, right_corner)
    if dist < 50 and dist_left > 40 and dist_bottom > 40 and dist_right > 40:
        x = top_corner[0]
        y = left_corner[1]
        w = right_corner[1] - y
        h = bottom_corner[0] - x
        cv2.rectangle(image, (y , x), (y+w, x+h ), (0, 255, 0), 3)
    to_choose = [x for x in range(diamond_corners.shape[0]) if x not in idx_to_remove]
    diamond_corners = diamond_corners[to_choose, :]




# find most left upper corner
# try to find the remaining corner
# if found remove all 4 corners from list
# if not found remove current corner point
# rectanlges =
# MAX_DIST = 20
# while(np.size(result) != 0):open_rectangle
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

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.bilateralFilter(gray, 11, 17, 17)
# edged = cv2.Canny(gray, 30, 200)

# cv2.imshow("Canny", edged)


# cv2.imshow('foo', dst)

# Threshold for an optimal value, it may vary depending on the image.
# image[dst>0.01*dst.max()]=[0,0,255]

for r in coners_copy:
    image[r[0], r[1]] = [0,0,255]

cv2.imshow('dst',image)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()