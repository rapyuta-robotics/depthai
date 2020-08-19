import os
import cv2
import numpy as np

image_size = (1280, 720)
intrinsics_file = "intrinsics.npz"
extrinsics_file = "extrinsics.npz"

image_folder = '/home/kota/dat/cnn/training/via-2.0.10/images/20200811_wooden'


class Rectifier:
    def __init__(self, intrinsics_file, extrinsics_file):
        self.intrinsics = np.load(intrinsics_file)
        self.extrinsics = np.load(extrinsics_file)

        self.map11, self.map12 = cv2.initUndistortRectifyMap(self.intrinsics['M1'], self.intrinsics['D1'],
                self.extrinsics['R1'], self.extrinsics['P1'], image_size, cv2.CV_16SC2)
        self.map21, self.map22 = cv2.initUndistortRectifyMap(self.intrinsics['M2'], self.intrinsics['D2'],
                self.extrinsics['R2'], self.extrinsics['P2'], image_size, cv2.CV_16SC2)

    def remap(self, img_l, img_r):
        img_l_new = cv2.remap(img_l, self.map11, self.map12, cv2.INTER_LINEAR)
        img_r_new = cv2.remap(img_r, self.map21, self.map22, cv2.INTER_LINEAR)
        return img_l_new, img_r_new


if __name__ == "__main__":
    rectifier = Rectifier(intrinsics_file, extrinsics_file)

    for i in range(100):
        left_path = os.path.join(image_folder, "left", "left_" + str(i).zfill(4) + ".png")
        right_path = os.path.join(image_folder, "right", "right_" + str(i).zfill(4) + ".png")
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)
        img_l, img_r = rectifier.remap(img_l, img_r)
        cv2.imshow("left", img_l)
        cv2.imshow("right", img_r)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
