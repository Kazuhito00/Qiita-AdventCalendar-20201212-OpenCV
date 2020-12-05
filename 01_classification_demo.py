#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv
import tensorflow as tf
import numpy as np

import json

from utils import CvFpsCalc
from utils import CvDrawText


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
    detection_count,
    classification_string,
    display_fps,
    trim_point,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # フォント
    font_path = './utils/font/KosugiMaru-Regular.ttf'

    # 四隅枠表示
    if detection_count < 4:
        gap_length = int((trim_point[2] - trim_point[0]) / 10) * 9
        cv.line(image, (trim_point[0], trim_point[1]),
                (trim_point[2] - gap_length, trim_point[1]), (255, 255, 255),
                3)
        cv.line(image, (trim_point[0] + gap_length, trim_point[1]),
                (trim_point[2], trim_point[1]), (255, 255, 255), 3)
        cv.line(image, (trim_point[2], trim_point[1]),
                (trim_point[2], trim_point[3] - gap_length), (255, 255, 255),
                3)
        cv.line(image, (trim_point[2], trim_point[1] + gap_length),
                (trim_point[2], trim_point[3]), (255, 255, 255), 3)
        cv.line(image, (trim_point[0], trim_point[3]),
                (trim_point[2] - gap_length, trim_point[3]), (255, 255, 255),
                3)
        cv.line(image, (trim_point[0] + gap_length, trim_point[3]),
                (trim_point[2], trim_point[3]), (255, 255, 255), 3)
        cv.line(image, (trim_point[0], trim_point[1]),
                (trim_point[0], trim_point[3] - gap_length), (255, 255, 255),
                3)
        cv.line(image, (trim_point[0], trim_point[1] + gap_length),
                (trim_point[0], trim_point[3]), (255, 255, 255), 3)

    line_x1 = int(image_width / 1.55)
    line_x2 = int(image_width / 1.1)
    line_y = int(image_height / 5)

    # 回転丸表示
    if detection_count > 0:
        draw_angle = int(detection_count * 45)
        cv.ellipse(image, (int(image_width / 2), int(image_height / 2)),
                   (10, 10), -45, 0, draw_angle, (255, 255, 255), -1)
    # 斜線表示
    if detection_count > 10:
        cv.line(image, (int(image_width / 2), int(image_height / 2)),
                (line_x1, line_y), (255, 255, 255), 2)

    # 横線・分類名・スコア表示
    if detection_count > 10:
        font_size = 32
        cv.line(image, (line_x1, line_y), (line_x2, line_y), (255, 255, 255),
                2)
        image = CvDrawText.puttext(
            image, classification_string,
            (line_x1 + 10, line_y - int(font_size * 1.25)), font_path,
            font_size, (255, 255, 255))

    # FPS描画 #######################################################
    fps_string = u"FPS:" + str(display_fps)
    image = CvDrawText.puttext(image, fps_string, (30, 30), font_path, 32,
                               (255, 255, 255))

    return image


def main():
    print("Image Classification Start...\n")

    # カメラ準備 ##############################################################
    cap = cv.VideoCapture(0)
    frame_width = 960
    frame_height = 540
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # ImageNet 日本語ラベル ###################################################
    jsonfile = open('utils/imagenet_class_index.json',
                    'r',
                    encoding="utf-8_sig")
    imagenet_ja_labels = json.load(jsonfile)
    imagenet_ja_label = {}
    for label_temp in imagenet_ja_labels:
        imagenet_ja_label[label_temp['num']] = label_temp['ja']

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    # モデルロード ############################################################
    model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )

    # メインループ #############################################################
    detection_count = 0
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
        trim_x1 = int(frame_width / 4) + 50
        trim_x2 = int(frame_width / 4 * 3) - 50
        trim_y1 = int(frame_height / 8)
        trim_y2 = int(frame_height / 8 * 7)
        trimming_image = debug_image[trim_y1:trim_y2, trim_x1:trim_x2]

        classifications = run_classify(model, trimming_image)

        # デバッグ情報描画 ####################################################
        # 表示名作成
        classification_string = ""
        for classification in classifications:
            if float(classification[2]) > 0.5:
                detection_count += 1

                classification_string = imagenet_ja_label[
                    classification[0]] + ":" + str('{:.1f}'.format(
                        float(classification[2]) * 100)) + "%"
            else:
                detection_count = 0
            break  # 1件のみ

        # 描画
        debug_image = draw_demo_image(
            debug_image,
            detection_count,
            classification_string,
            display_fps,
            (trim_x1, trim_y1, trim_x2, trim_y2),
        )

        # 画面反映 #######################################################
        cv.imshow('Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
