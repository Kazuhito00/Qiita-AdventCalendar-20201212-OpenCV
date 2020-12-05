#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv
import numpy as np
import tensorflow.compat.v1 as tf

from utils import CvFpsCalc
from utils import CvDrawText


def graph_load(path):
    graph = tf.Graph()
    graph_def = None
    with open(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    sess = tf.Session(graph=graph)

    return sess


def session_run(sess, image, inf_size=(480, 320)):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    temp_image = copy.deepcopy(image)
    temp_image = cv.resize(temp_image, inf_size)
    batch_seg_map = sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: [np.asarray(temp_image)]})
    seg_map = batch_seg_map[0]

    return seg_map


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)

    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    colormap[15] = [0, 0, 0]

    return colormap


def create_pascal_label_personmask():
    colormap = np.zeros((256, 3), dtype=int)

    colormap[15] = [255, 255, 255]

    return colormap


def label_to_color_image(label):
    colormap = create_pascal_label_colormap()

    return colormap[label]


def label_to_person_mask(label):
    colormap = create_pascal_label_personmask()

    return colormap[label]


def draw_demo_image(
        image,
        segmentation_map,
        display_fps,
        inf_size=(480, 320),
):
    # フォント
    font_path = './utils/font/x12y20pxScanLine.ttf'

    # ピクセル塗りつぶし
    image_width, image_height = image.shape[1], image.shape[0]

    draw_image = copy.deepcopy(image)
    draw_image = cv.resize(draw_image, inf_size)

    seg_image = label_to_color_image(segmentation_map).astype(np.uint8)
    seg_mask = label_to_person_mask(segmentation_map).astype(np.uint8)

    draw_image = np.where(seg_mask == 255, seg_image, draw_image)

    draw_image = cv.resize(draw_image, (image_width, image_height))

    # FPS描画
    fps_string = u"FPS:" + str(display_fps)
    draw_image = CvDrawText.puttext(draw_image, fps_string, (15, 15),
                                    font_path, 32, (0, 0, 0))
    return draw_image


def main():
    print("Semantic Segmentation Start...\n")

    # カメラ準備 ##############################################################
    cap = cv.VideoCapture(0)
    frame_width = 960
    frame_height = 540
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    # モデルロード ############################################################
    frozen_path = "model/deeplab_v3/deeplabv3_mnv2.pb"
    sess = graph_load(frozen_path)

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
        segmentation_map = session_run(sess, debug_image)

        # デバッグ情報描画 ####################################################
        debug_image = draw_demo_image(
            debug_image,
            segmentation_map,
            display_fps,
        )

        # 画面反映 #######################################################
        cv.imshow('Demo', debug_image)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break


if __name__ == '__main__':
    main()
