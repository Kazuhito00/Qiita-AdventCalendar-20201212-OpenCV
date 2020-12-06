#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import copy
import cv2 as cv

try:
    import tensorflow.compat.v1 as tf
except Exception:
    import tensorflow as tf

from utils import *

# バウンディングボックスリスト ##################################################
bba_function = [
    [bba_rotate_dotted_ring3, None],
    [bba_translucent_rectangle, None],
    [bba_translucent_circle, None],
    [bba_look_into_the_muzzle, None],
    [bba_translucent_rectangle_fill1, u'HAND'],
    [bba_square_obit, None],
    [bba_annotation_line, u'検出結果：手'],
    [bba_ground_glass, None],
]


def graph_load(path):
    config = tf.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
        allow_growth=True))

    with tf.compat.v1.Graph().as_default() as net_graph:
        graph_data = tf.gfile.FastGFile(path, 'rb').read()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(graph_data)
        tf.import_graph_def(graph_def, name='')

    sess = tf.compat.v1.Session(graph=net_graph, config=config)
    sess.graph.as_default()

    return sess


def session_run(sess, inp):
    out = sess.run(
        [
            sess.graph.get_tensor_by_name('num_detections:0'),
            sess.graph.get_tensor_by_name('detection_scores:0'),
            sess.graph.get_tensor_by_name('detection_boxes:0'),
            sess.graph.get_tensor_by_name('detection_classes:0')
        ],
        feed_dict={
            'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)
        },
    )
    return out


def main():
    print("Object Detection Start...\n")

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(0)
    frame_width = 960
    frame_height = 540
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

    # 手検出モデルロード #######################################################
    sess = graph_load('model/mobilenetv2_ssd_lite/hand_detection.pb')

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc()

    index = 0
    fps = 15
    animation_count = 0

    while True:
        start_time = time.time()
        animation_count += 1

        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        frame_width, frame_height = frame.shape[1], frame.shape[0]
        debug_image = copy.deepcopy(frame)

        # 手検出実施 ###########################################################
        inp = cv.resize(frame, (512, 512))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = session_run(sess, inp)

        num_detections = int(out[0][0])
        for i in range(num_detections):
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            class_id = int(out[3][0][i])

            if score < 0.8:
                continue

            # 手検出結果可視化 #################################################
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)

            debug_image = bba_function[index][0](
                image=debug_image,
                p1=(x1, y1),
                p2=(x2, y2),
                text=bba_function[index][1],
                fps=fps,
                animation_count=animation_count,
            )

        # 画面反映 #############################################################
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv.LINE_AA)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, (252, 244, 234), 1,
                   cv.LINE_AA)
        # cv.putText(debug_image,
        #            str(bba_function[index][0].__name__) + '()', (10, 50),
        #            cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
        cv.imshow('Demo', debug_image)

        # キー処理(N：次へ、P：前へ、ESC：終了) #################################
        key = cv.waitKey(1)
        if key == 110:  # N
            index = 0 if ((index + 1) >= len(bba_function)) else (index + 1)
        if key == 112:  # P
            index = len(bba_function) - 1 if ((index - 1) < 0) else (index - 1)
        if key == 27:  # ESC
            break

        # FPS調整 #############################################################
        elapsed_time = time.time() - start_time
        sleep_time = max(0, ((1.0 / fps) - elapsed_time))
        time.sleep(sleep_time)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
