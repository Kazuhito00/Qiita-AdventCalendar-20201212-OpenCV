#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import cv2 as cv


def bba_look_into_the_muzzle_mask(
        image,
        p1,
        p2,
        color=None,  # unused
        thickness=None,  # unused
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=None,  # unused
        mask_image=None,
):
    image_width, image_height = image.shape[1], image.shape[0]

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    if mask_image is None:
        mask_image = np.zeros((image_height, image_width, 3), np.uint8)

    cv.circle(
        mask_image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
        int((y2 - y1) * (1 / 2)), (255, 255, 255),
        thickness=-1)

    return mask_image


def bba_look_into_the_muzzle_fix(image, mask_image):
    draw_image = copy.deepcopy(image)
    draw_image = cv.bitwise_and(draw_image, mask_image)
    return draw_image


def bba_look_into_the_muzzle(
        image,
        p1,
        p2,
        color=None,  # unused
        thickness=None,  # unused
        font=None,  # unused
        text=None,  # unused
        fps=None,  # unused
        animation_count=None,  # unused
):

    draw_image = copy.deepcopy(image)

    mask_image = bba_look_into_the_muzzle_mask(
        image, p1, p2, color, thickness, font, text, fps, animation_count)
    draw_image = bba_look_into_the_muzzle_fix(draw_image, mask_image)

    return draw_image
