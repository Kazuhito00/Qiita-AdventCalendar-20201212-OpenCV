#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import copy
from collections import deque

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import CvFpsCalc
from utils import CvDrawText
from utils import CvOverlayImage


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--model",
                        default='model/efficient_det_fingerframe/saved_model')
    parser.add_argument("--score_th", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    fps = args.fps

    model_path = args.model
    score_th = args.score_th

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    video = cv.VideoCapture('utils/map.mp4')

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    # モデルロード #############################################################
    DEFAULT_FUNCTION_KEY = 'serving_default'
    loaded_model = tf.saved_model.load(model_path)
    inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

    buffer_len = 5
    deque_x1 = deque(maxlen=buffer_len)
    deque_y1 = deque(maxlen=buffer_len)
    deque_x2 = deque(maxlen=buffer_len)
    deque_y2 = deque(maxlen=buffer_len)

    # フォント
    font_path = './utils/font/x12y20pxScanLine.ttf'

    while True:
        # FPS算出 #############################################################
        display_fps = cvFpsCalc.get()

        start_time = time.time()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 検出実施 #############################################################
        frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
        image_np_expanded = np.expand_dims(frame, axis=0)

        output = run_inference_single_image(image_np_expanded, inference_func)

        num_detections = output['num_detections']
        for i in range(num_detections):
            score = output['detection_scores'][i]
            bbox = output['detection_boxes'][i]
            # class_id = output['detection_classes'][i].astype(np.int)

            if score < score_th:
                continue

            # 検出結果可視化 ###################################################
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            risize_ratio = 0.15
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            x1 = x1 + int(bbox_width * risize_ratio)
            y1 = y1 + int(bbox_height * risize_ratio)
            x2 = x2 - int(bbox_width * risize_ratio)
            y2 = y2 - int(bbox_height * risize_ratio)

            x1 = int((x1 - 5) / 10) * 10
            y1 = int((y1 - 5) / 10) * 10
            x2 = int((x2 + 5) / 10) * 10
            y2 = int((y2 + 5) / 10) * 10

            deque_x1.append(x1)
            deque_y1.append(y1)
            deque_x2.append(x2)
            deque_y2.append(y2)
            x1 = int(sum(deque_x1) / len(deque_x1))
            y1 = int(sum(deque_y1) / len(deque_y1))
            x2 = int(sum(deque_x2) / len(deque_x2))
            y2 = int(sum(deque_y2) / len(deque_y2))

            ret, video_frame = video.read()
            if ret is not False:
                video.grab()
                video.grab()

                debug_add_image = np.zeros((frame_height, frame_width, 3),
                                           np.uint8)
                map_resize_image = cv.resize(video_frame,
                                             ((x2 - x1), (y2 - y1)))
                debug_add_image = CvOverlayImage.overlay(
                    debug_add_image, map_resize_image, (x1, y1))
                debug_add_image = cv.cvtColor(debug_add_image,
                                              cv.COLOR_BGRA2BGR)
                # cv.imshow('1', debug_add_image)
                debug_image = cv.addWeighted(debug_image, 1.0, debug_add_image,
                                             2.0, 0)
            else:
                video = cv.VideoCapture('map.mp4')

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

        # 画面反映 #############################################################
        fps_string = u"FPS:" + str(display_fps)
        debug_image = CvDrawText.puttext(debug_image, fps_string, (17, 17),
                                         font_path, 32, (255, 255, 255))
        fps_string = u"FPS:" + str(display_fps)
        debug_image = CvDrawText.puttext(debug_image, fps_string, (15, 15),
                                         font_path, 32, (0, 56, 86))
        cv.imshow('Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
