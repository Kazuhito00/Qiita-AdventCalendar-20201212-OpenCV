#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv


def bba_square_obit(
        image,
        p1,
        p2,
        color=(255, 255, 255),
        thickness=1,
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=0,
):

    draw_image = copy.deepcopy(image)

    reduction_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    rectangle_width = x2 - x1
    rectangle_height = y2 - y1

    position_offsets = [
        [0, 0],
        [int(rectangle_width * (1 / 5)), 0],
        [int(rectangle_width * (2 / 5)), 0],
        [int(rectangle_width * (3 / 5)), 0],
        [int(rectangle_width * (4 / 5)), 0],
        [int(rectangle_width * (5 / 5)), 0],
        [int(rectangle_width * (5 / 5)),
         int(rectangle_height * (1 / 5))],
        [int(rectangle_width * (5 / 5)),
         int(rectangle_height * (2 / 5))],
        [int(rectangle_width * (5 / 5)),
         int(rectangle_height * (3 / 5))],
        [int(rectangle_width * (5 / 5)),
         int(rectangle_height * (4 / 5))],
        [int(rectangle_width * (5 / 5)),
         int(rectangle_height * (5 / 5))],
        [int(rectangle_width * (4 / 5)),
         int(rectangle_height * (5 / 5))],
        [int(rectangle_width * (3 / 5)),
         int(rectangle_height * (5 / 5))],
        [int(rectangle_width * (2 / 5)),
         int(rectangle_height * (5 / 5))],
        [int(rectangle_width * (1 / 5)),
         int(rectangle_height * (5 / 5))],
        [0, int(rectangle_height * (5 / 5))],
        [0, int(rectangle_height * (4 / 5))],
        [0, int(rectangle_height * (3 / 5))],
        [0, int(rectangle_height * (2 / 5))],
        [0, int(rectangle_height * (1 / 5))],
    ]

    for index, offset in enumerate(position_offsets):
        square_harf_width = int(
            rectangle_width /
            reduction_list[(animation_count - index) % len(reduction_list)])

        cv.rectangle(
            draw_image,
            (x1 + offset[0] - square_harf_width,
             y1 + offset[1] - square_harf_width),
            (x1 + offset[0] + square_harf_width,
             y1 + offset[1] + square_harf_width),
            color,
            thickness=thickness,
        )

    return draw_image
