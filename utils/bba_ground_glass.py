#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv


def bba_ground_glass(
        image,
        p1,
        p2,
        color=(255, 255, 255),
        thickness=1,
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=None,  # unused
):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    rectangle_width = x2 - x1
    rectangle_height = y2 - y1

    draw_image = copy.deepcopy(image)
    bbox_area = draw_image[y1:y2, x1:x2]

    bbox_area = cv.GaussianBlur(bbox_area, (51, 51), 50)
    draw_image[y1:rectangle_height + y1, x1:rectangle_width + x1] = bbox_area

    cv.rectangle(
        draw_image,
        (x1 + int(rectangle_width / 20), y1 + int(rectangle_width / 20)),
        (x2 - int(rectangle_width / 20), y2 - int(rectangle_width / 20)),
        color,
        thickness=thickness)
    cv.rectangle(draw_image, (x1, y1), (x2, y2), color, thickness=thickness)

    return draw_image
