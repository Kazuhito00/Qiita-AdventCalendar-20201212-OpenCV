#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv
import tensorflow as tf
import numpy as np

import utils.cvui as cvui
from utils import CvFpsCalc


def run_classify(model, image, top_num=5):
    """
    [summary]
        画像クラス分類
    Parameters
    ----------
    model : model
        クラス分類用モデル
    image : image
        推論対象の画像
    None
    """
    inp = cv.resize(image, (224, 224))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    inp = np.expand_dims(inp, axis=0)
    tensor = tf.convert_to_tensor(inp)
    tensor = tf.keras.applications.efficientnet.preprocess_input(tensor)

    classifications = model.predict(tensor)

    classifications = tf.keras.applications.efficientnet.decode_predictions(
        classifications,
        top=top_num,
    )
    classifications = np.squeeze(classifications)
    return classifications


def draw_demo_image(
    image,
    classifications,
    display_fps,
):
    image_width, image_height = image.shape[1], image.shape[0]

    cvuiframe = np.zeros((image_height + 6, image_width + 6 + 200, 3),
                         np.uint8)
    cvuiframe[:] = (49, 52, 49)

    # 画像：撮影映像
    display_frame = copy.deepcopy(image)
    cvui.image(cvuiframe, 3, 3, display_frame)

    # 文字列：FPS
    cvui.printf(cvuiframe, image_width + 15, 15, 0.4, 0xFFFFFF,
                'FPS : ' + str(display_fps))

    # 文字列、バー：クラス分類結果
    if classifications is not None:
        for i, classification in enumerate(classifications):
            cvui.printf(cvuiframe, image_width + 15,
                        int(image_height / 4) + (i * 40), 0.4, 0xFFFFFF,
                        classification[1])
            cvui.rect(cvuiframe, image_width + 15,
                      int(image_height / 4) + 15 + (i * 40),
                      int(181 * float(classification[2])), 12, 0xFFFFFF,
                      0xFFFFFF)

    return cvuiframe


def main():
    print("Image Classification Start...\n")

    # カメラ準備 ##############################################################
    cap = cv.VideoCapture(0)
    frame_width = 960
    frame_height = 540
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    # モデルロード ############################################################
    model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )

    # CVUI初期化 ##############################################################
    cvui.init("Demo")

    # メインループ #############################################################
    while True:
        # FPS算出 #############################################################
        display_fps = cvFpsCalc.get()
        if display_fps == 0:
            display_fps = 0.01

        # カメラキャプチャ ####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        debug_image = copy.deepcopy(frame)

        # 検出実施 ############################################################
        trim_x1 = int((frame_width - frame_height) / 2)
        trim_x2 = frame_width - int((frame_width - frame_height) / 2)
        trimming_image = debug_image[0:frame_height, trim_x1:trim_x2]

        classifications = run_classify(model, trimming_image, 10)

        # デバッグ情報描画 ####################################################
        debug_image = draw_demo_image(
            trimming_image,
            classifications,
            display_fps,
        )

        # 画面反映 #######################################################
        cvui.imshow('Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
