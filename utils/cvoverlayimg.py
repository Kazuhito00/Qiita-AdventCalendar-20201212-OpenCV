#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=I0011,E1101,E0401

import copy

import cv2 as cv
import numpy as np
from PIL import Image


class CvOverlayImage(object):
    """
    [summary]
      OpenCV形式の画像に指定画像を重ねる
    """
    def __init__(self):
        pass

    @classmethod
    def overlay(
        cls,
        cv_background_image,
        cv_overlay_image,
        point,
    ):
        """
        [summary]
          OpenCV形式の画像に指定画像を重ねる
        Parameters
        ----------
        cv_background_image : [OpenCV Image]
        cv_overlay_image : [OpenCV Image]
        point : [(x, y)]
        Returns : [OpenCV Image]
        """
        # 結果格納用データ
        cv_result_image = copy.deepcopy(cv_background_image)

        # 描画位置計算
        x, y = point
        overlay_h, overlay_w = cv_overlay_image.shape[:2]
        background_h, background_w = cv_result_image.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2 = min(x + overlay_w, background_w)
        y2 = min(y + overlay_h, background_h)

        # 画像外の座標の場合はバックグラウンド画像をそのまま返却
        if not ((-overlay_w < x < background_w) and
                (-overlay_h < y < background_h)):
            return cv_result_image

        # 画像重ね合わせ
        pil_background_roi = Image.fromarray(cv_result_image[y1:y2, x1:x2])
        pil_overlay_image = Image.fromarray(cv_overlay_image[y1 - y:y2 - y,
                                                             x1 - x:x2 - x])
        pil_background_roi.paste(pil_overlay_image, (0, 0), pil_overlay_image)
        background_roi = np.array(pil_background_roi, dtype=np.uint8)
        cv_result_image[y1:y2, x1:x2] = background_roi

        return cv_result_image
