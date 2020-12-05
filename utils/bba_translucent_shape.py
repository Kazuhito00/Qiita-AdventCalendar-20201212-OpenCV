#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import cv2 as cv

from utils import CvDrawText


def bba_translucent_rectangle(
        image,
        p1,
        p2,
        color=(0, 64, 0),
        thickness=5,
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=None,  # unused
):

    draw_image = copy.deepcopy(image)
    image_width, image_height = draw_image.shape[1], draw_image.shape[0]

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    draw_add_image = np.zeros((image_height, image_width, 3), np.uint8)
    cv.rectangle(draw_add_image, (x1, y1), (x2, y2),
                 color,
                 thickness=thickness)
    draw_image = cv.add(draw_image, draw_add_image)

    return draw_image


def bba_translucent_rectangle_fill1(
        image,
        p1,
        p2,
        color=(0, 64, 0),
        thickness=None,  # unused
        font='utils/font/x12y20pxScanLine.ttf',
        text=None,
        fps=None,  # unused
        animation_count=None,  # unused
):

    draw_image = copy.deepcopy(image)
    image_width, image_height = draw_image.shape[1], draw_image.shape[0]

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    rectangle_width = x2 - x1
    rectangle_height = y2 - y1

    font_size = int((y2 - y1) * (2 / 9))

    draw_add_image = np.zeros((image_height, image_width, 3), np.uint8)
    cv.rectangle(draw_add_image, (x1, y1), (x2, y2), color, thickness=10)
    cv.rectangle(
        draw_add_image,
        (x1 + int(rectangle_width / 20), y1 + int(rectangle_height / 20)),
        (x2 - int(rectangle_width / 20), y2 - int(rectangle_height / 20)),
        color,
        thickness=-1)
    if (font is not None) and (text is not None):
        draw_add_image = CvDrawText.puttext(
            draw_add_image, text,
            (int(((x1 + x2) / 2) -
                 (font_size / 0.9)), int(((y1 + y2) / 2) - (font_size / 2))),
            font, font_size, (0, 0, 0))
    draw_image = cv.add(draw_image, draw_add_image)

    return draw_image


def bba_translucent_circle(
        image,
        p1,
        p2,
        color=(128, 0, 0),
        thickness=5,
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=None,  # unused
):

    draw_image = copy.deepcopy(image)
    image_width, image_height = draw_image.shape[1], draw_image.shape[0]

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    draw_add_image = np.zeros((image_height, image_width, 3), np.uint8)
    cv.circle(draw_add_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
              int((y2 - y1) * (1 / 2)),
              color,
              thickness=thickness)
    draw_image = cv.add(draw_image, draw_add_image)

    return draw_image
