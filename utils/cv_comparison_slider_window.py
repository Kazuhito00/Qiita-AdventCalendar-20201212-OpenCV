import numpy as np
import cv2 as cv


class CvComparisonSliderWindow(object):
    def __init__(self,
                 window_name='debug',
                 line_color=(255, 255, 255),
                 line_thickness=0):
        self.window_name = window_name
        self.click_point = [1, 1]
        self.line_color = line_color
        self.line_thickness = line_thickness

        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            self.click_point = [x, y]

    def imshow(self, image1, image2, fps=None):
        image1_width, image1_height = image1.shape[1], image1.shape[0]
        image2_width, image2_height = image2.shape[1], image2.shape[0]
        if ((image1_width != image2_width)
                or (image1_height != image2_height)):
            image2 = cv.resize(image2, (image1_width, image1_height))

        image_height = image1.shape[0]

        crop_image1 = image1[:, 0:self.click_point[0]]
        crop_image2 = image2[:, self.click_point[0] + 1:]
        concat_image = np.concatenate([crop_image1, crop_image2], axis=1)

        if self.line_thickness > 0:
            cv.line(concat_image, (self.click_point[0], 1),
                    (self.click_point[0], image_height),
                    self.line_color,
                    thickness=self.line_thickness)

        if fps is not None:
            cv.putText(concat_image, "FPS:" + str(fps), (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv.LINE_AA)
            cv.putText(concat_image, "FPS:" + str(fps), (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1.3, (252, 244, 234), 1,
                       cv.LINE_AA)

        cv.imshow(self.window_name, concat_image)

        return
