#!/usr/bin/env python3

import os

import platform
from contextlib import contextmanager

import numpy as np
import cv2

import depthai
from depthai_helpers import camera_info
from depthai_helpers.config_manager import BlobManager

# from depthai_helpers.version_check import check_depthai_version
# check_depthai_version()

on_embedded = platform.machine().startswith('arm') \
    or platform.machine().startswith('aarch64')

show_size = (640, 360)

chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
subpix_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.05)


class CalibMegaAI():
    def __init__(self):

        blob_man_args = {
                        'cnn_model': 'mobilenet-ssd',
                        'cnn_model2': '',
                        'model_compilation_target': 'auto'
                    }
        shaves = 7
        cmx_slices = 7
        NN_engines = 1
        blobMan = BlobManager(blob_man_args, True, shaves, cmx_slices, NN_engines)

        self.config = {
            'streams':
                ['video'] if not on_embedded else
                [{'name': 'video', "max_fps": 10.0}],
            'ai':
                {
                    'blob_file': blobMan.blob_file,
                    'blob_file_config': blobMan.blob_file_config,
                    'shaves': shaves,
                    'cmx_slices': cmx_slices,
                    'NN_engines': NN_engines,
                },

            'video_config':
                {
                    'profile': 'mjpeg',
                    'quality': 95
                },
            'camera':
                {
                    'rgb':
                    {
                        # 3840x2160, 1920x1080
                        # only UHD/1080p/30 fps supported for now
                        'resolution_h': 1080,
                        'fps': 30.0,
                    },
                },
        }

        self.objp = np.zeros((9 * 6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        self.objpoints = []
        self.imgpoints = []

        self.outfolder = "dataset/megaAI/"
        if not os.path.exists(self.outfolder):
            os.makedirs(self.outfolder)

    @contextmanager
    def get_pipeline(self):
        # Possible to try and reboot?
        # The following doesn't work (have to manually hit switch on device)
        # depthai.reboot_device
        # time.sleep(1)

        pipeline = None

        try:
            self.device = depthai.Device("", False)
            pipeline = self.device.create_pipeline(self.config)
            self.device.request_af_mode(depthai.AutofocusMode.AF_MODE_EDOF)
        except RuntimeError:
            raise RuntimeError("Unable to initialize device. Try to reset it")

        if pipeline is None:
            raise RuntimeError("Unable to create pipeline")

        try:
            yield pipeline
        finally:
            del pipeline

    def run(self):
        count = 0
        with self.get_pipeline() as pipeline:
            capture = False
            while True:
                key = cv2.waitKey(1)
                if key == ord(" "):
                    capture = True
                    # self.device.request_jpeg()
                elif key == 27 or key == ord("q"):
                    break

                _, data_list = pipeline.get_available_nnet_and_data_packets()
                if not any([d.stream_name == 'video' for d in data_list]):
                    continue

                for packet in data_list:
                    if packet.stream_name == 'video':
                        img = cv2.imdecode(packet.getData(), cv2.IMREAD_COLOR)

                img_small = cv2.resize(img, show_size)
                cv2.imshow("video", img_small)

                if capture:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(
                        gray, (9, 6), chessboard_flags)

                    if ret:
                        corners_sub = cv2.cornerSubPix(
                            gray, corners, (15, 15), (-1, -1), subpix_criteria)

                        self.imgpoints.append(corners_sub)
                        self.objpoints.append(self.objp)

                        cv2.imwrite(self.outfolder + "video_%04d.png" % count, img)
                        count += 1

                    img_draw = img.copy()
                    cv2.drawChessboardCorners(img_draw, (9, 6), corners, ret)
                    cv2.imshow("corners", img_draw)
                    cv2.waitKey(0)
                    cv2.destroyWindow("corners")

                    capture = False

            cv2.destroyAllWindows()

            if len(self.objpoints) == 0:
                print("No corner points were captured.")
                return

        cal_data = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1],
                                       None, None)  # ret, M, d, r, t

        print("Using {} images, error: {}".format(count, cal_data[0]))
        camera_info.write_camera_info("camera_info_maga.yaml",
                                      "rgb", (1920, 1080),
                                      cal_data[1], cal_data[2])


if __name__ == "__main__":
    CalibMegaAI().run()
