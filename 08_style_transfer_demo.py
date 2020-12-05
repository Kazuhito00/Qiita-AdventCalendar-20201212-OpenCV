#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import cv2 as cv
import numpy as np
import tensorflow.compat.v1 as tf

from utils import CvFpsCalc
import utils.white_box_cartoonization as wbc_network

from utils.cv_comparison_slider_window import CvComparisonSliderWindow


def graph_load(path):
    tf.disable_eager_execution()

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = wbc_network.unet_generator(input_photo)
    final_out = wbc_network.guided_filter(input_photo,
                                          network_out,
                                          r=1,
                                          eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(path))

    return sess, input_photo, final_out


def session_run(sess, image, input_photo, final_out):
    debug_image = resize_crop(image)

    batch_image = debug_image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)

    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def main():
    print("Style Transfer Start...\n")

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(0)
    frame_width = 960
    frame_height = 540
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # モデルロード #############################################################
    sess, input_photo, final_out = graph_load('model/white_box_cartoonization')

    # CvComparisonSliderWindow準備 ############################################
    cvwindow = CvComparisonSliderWindow(
        window_name='Demo',
        line_color=(255, 255, 255),
        line_thickness=1,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 変換実施 #############################################################
        out = session_run(sess, debug_image, input_photo, final_out)

        # 画面反映 #############################################################
        cvwindow.imshow(frame, out, fps=display_fps)

        # キー処理 #############################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


if __name__ == '__main__':
    main()
