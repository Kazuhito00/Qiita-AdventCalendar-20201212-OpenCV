#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv


def bba_rotate_dotted_ring3(
        image,
        p1,
        p2,
        color=(255, 255, 205),
        thickness=None,  # unused
        font=None,  # unused
        text=None,  # unused
        fps=10,
        animation_count=0,
):

    draw_image = copy.deepcopy(image)

    animation_count = int(135 / fps) * animation_count

    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    radius = int((y2 - y1) * (5 / 10))
    ring_thickness = int(radius / 20)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 0 + animation_count, 0, 50, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 80 + animation_count, 0, 50, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 150 + animation_count, 0, 30, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 200 + animation_count, 0, 10, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 230 + animation_count, 0, 10, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 260 + animation_count, 0, 60, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 337 + animation_count, 0, 5, color,
               ring_thickness)

    radius = int((y2 - y1) * (4.5 / 10))
    ring_thickness = int(radius / 10)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 0 - animation_count, 0, 50, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 80 - animation_count, 0, 50, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 150 - animation_count, 0, 30, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 200 - animation_count, 0, 30, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 260 - animation_count, 0, 60, color,
               ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 337 - animation_count, 0, 5, color,
               ring_thickness)

    radius = int((y2 - y1) * (4 / 10))
    ring_thickness = int(radius / 15)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 30 + int(animation_count / 3 * 2),
               0, 50, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 110 + int(animation_count / 3 * 2),
               0, 50, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 180 + int(animation_count / 3 * 2),
               0, 30, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 230 + int(animation_count / 3 * 2),
               0, 10, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 260 + int(animation_count / 3 * 2),
               0, 10, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 290 + int(animation_count / 3 * 2),
               0, 60, color, ring_thickness)
    cv.ellipse(draw_image, (int((x1 + x2) / 2), int(
        (y1 + y2) / 2)), (radius, radius), 367 + int(animation_count / 3 * 2),
               0, 5, color, ring_thickness)

    return draw_image
