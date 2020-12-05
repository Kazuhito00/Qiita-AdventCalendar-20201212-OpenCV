#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv

from utils import CvDrawText


def bba_black_ring_wa(
        image,
        p1,
        p2,
        color=(185, 0, 0),
        thickness=None,  # unused
        font='utils/font/衡山毛筆フォント.ttf',
        text=None,
        fps=None,  # unused
        animation_count=None,  # unused
):

    draw_image = copy.deepcopy(image)

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    font_size = int((y2 - y1) * (4 / 5))

    cv.circle(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
              int((y2 - y1) * (3 / 7)), (255, 255, 255), 20)
    cv.circle(draw_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
              int((y2 - y1) * (3 / 7)), (0, 0, 0), 10)

    if (font is not None) and (text is not None):
        draw_image = CvDrawText.puttext(
            draw_image, text,
            (int(((x1 + x2) / 2) -
                 (font_size / 2)), int(((y1 + y2) / 2) - (font_size / 2))),
            font, font_size, color)

    return draw_image
