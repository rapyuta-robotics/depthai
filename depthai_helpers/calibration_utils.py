import cv2
import glob
import os
import shutil
import numpy as np
import re
import time

# Creates a set of 13 polygon coordinates
def setPolygonCoordinates(height, width):
    horizontal_shift = width//4
    vertical_shift = height//4

    margin = 60
    slope = 150

    p_coordinates = [
            [[margin,0], [margin,height], [width//2, height-slope], [width//2, slope]],
            [[horizontal_shift, 0], [horizontal_shift, height], [width//2 + horizontal_shift, height-slope], [width//2 + horizontal_shift, slope]],
            [[horizontal_shift*2-margin, 0], [horizontal_shift*2-margin, height], [width//2 + horizontal_shift*2-margin, height-slope], [width//2 + horizontal_shift*2-margin, slope]],

            [[margin,margin], [margin, height-margin], [width-margin, height-margin], [width-margin, margin]],

            [[width-margin, 0], [width-margin, height], [width//2, height-slope], [width//2, slope]],
            [[width-horizontal_shift, 0], [width-horizontal_shift, height], [width//2-horizontal_shift, height-slope], [width//2-horizontal_shift, slope]],
            [[width-horizontal_shift*2+margin, 0], [width-horizontal_shift*2+margin, height], [width//2-horizontal_shift*2+margin, height-slope], [width//2-horizontal_shift*2+margin, slope]],

            [[0,margin], [width, margin], [width-slope, height//2], [slope, height//2]],
            [[0,vertical_shift], [width, vertical_shift], [width-slope, height//2+vertical_shift], [slope, height//2+vertical_shift]],
            [[0,vertical_shift*2-margin], [width, vertical_shift*2-margin], [width-slope, height//2+vertical_shift*2-margin], [slope, height//2+vertical_shift*2-margin]],

            [[0,height-margin], [width, height-margin], [width-slope, height//2], [slope, height//2]],
            [[0,height-vertical_shift], [width, height-vertical_shift], [width-slope, height//2-vertical_shift], [slope, height//2-vertical_shift]],
            [[0,height-vertical_shift*2+margin], [width, height-vertical_shift*2+margin], [width-slope, height//2-vertical_shift*2+margin], [slope, height//2-vertical_shift*2+margin]]
        ]
    return p_coordinates

def getPolygonCoordinates(idx, p_coordinates):
    return p_coordinates[idx]

def getNumOfPolygons(p_coordinates):
    return len(p_coordinates)

# Filters polygons to just those at the given indexes.
def select_polygon_coords(p_coordinates,indexes):
    if indexes == None:
        # The default
        return p_coordinates
    else:
        print("Filtering polygons to those at indexes=",indexes)
        return [p_coordinates[i] for i in indexes]

def image_filename(stream_name,polygon_index,total_num_of_captured_images):
    return "{stream_name}_p{polygon_index}_{total_num_of_captured_images}.png".format(stream_name=stream_name,polygon_index=polygon_index,total_num_of_captured_images=total_num_of_captured_images)

def polygon_from_image_name(image_name):
    """Returns the polygon index from an image name (ex: "left_p10_0.png" => 10)"""
    return int(re.findall("p(\d+)",image_name)[0])

class StereoCalibration(object):
    """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def __init__(self):
        """Class to Calculate Calibration and Rectify a Stereo Camera."""

    def calibrate(self, filepath, square_size, out_filepath, flags):
        """Function to calculate calibration for stereo camera."""
        start_time = time.time()
        # init object data
        self.objp = np.zeros((9 * 6, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for pt in self.objp:
            pt *= square_size

        # process images, detect corners, refine and save data
        self.process_images(filepath)

        # run calibration procedure and construct Homography
        self.stereo_calibrate()

        # save data to binary file
        self.H.tofile(out_filepath)

        # append specific flags to file
        with open(out_filepath, "ab") as fp:
            fp.write(bytearray(flags))

        print("Calibration file written to %s." % (out_filepath))
        print("\tTook %i seconds to run image processing." % (round(time.time() - start_time, 2)))
        # show debug output for visual inspection
        print("\nRectifying dataset for visual inspection")
        self.show_rectified_images(filepath, out_filepath)

    def process_images(self, filepath):
        """Read images, detect corners, refine corners, and save data."""
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        self.imgpoints_rgb = []  # 2d points in image plane.
        self.calib_successes_lr = [] # polygon ids of left/right image sets with checkerboard corners.
        self.calib_successes_rgb = [] # polygon ids of left/rgb image sets with checkerboard corners.

        images_left = glob.glob(filepath + "/left/*")
        images_right = glob.glob(filepath + "/right/*")
        images_rgb = glob.glob(filepath + "/jpegout/*")
        images_left.sort()
        images_right.sort()
        images_rgb.sort()

        print("\nAttempting to read images for left camera from dir: " +
              filepath + "/left/")
        print("Attempting to read images for right camera from dir: " +
              filepath + "/right/")
        print("Attempting to read images for right camera from dir: " +
              filepath + "/jpegout/")

        assert len(images_left) != 0, "ERROR: Images not read correctly, check directory"
        assert len(images_right) != 0, "ERROR: Images not read correctly, check directory"
        assert len(images_rgb) != 0, "ERROR: Images not read correctly, check directory"

        for image_left, image_right, image_rgb in zip(images_left, images_right, images_rgb):
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)
            img_rgb = cv2.imread(image_rgb)

            assert img_l is not None, "ERROR: Images not read correctly"
            assert img_r is not None, "ERROR: Images not read correctly"
            assert img_rgb is not None, "ERROR: Images not read correctly"

            print("Finding chessboard corners for %s and %s..." % (os.path.basename(image_left), os.path.basename(image_right)))
            start_time = time.time()

            # Find the chess board corners
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_l, corners_l = cv2.findChessboardCorners(img_l, (9, 6), flags)
            ret_r, corners_r = cv2.findChessboardCorners(img_r, (9, 6), flags)
            ret_rgb, corners_rgb = cv2.findChessboardCorners(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), (9, 6), flags)

            img_pt_l = cv2.drawChessboardCorners(np.dstack([img_l, img_l, img_l]), (9, 6), corners_l ,True)
            img_pt_r = cv2.drawChessboardCorners(np.dstack([img_r, img_r, img_r]), (9, 6), corners_r ,True)
            img_pt_rgb = cv2.drawChessboardCorners(img_rgb, (9, 6), corners_rgb ,True)
            img_pt_rgb = cv2.resize(img_pt_rgb, (img_pt_l.shape[1], img_pt_l.shape[0]))
            # cv2.imwrite(image_left.replace('.png', '_pts.png'), img_pt_l)
            # cv2.imwrite(image_right.replace('.png', '_pts.png'), img_pt_r)

            # cv2.imshow("points", np.hstack([img_pt_l, img_pt_r, img_pt_rgb]))
            # cv2.waitKey(0)
            # cv2.destroyWindow("points")

            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                             cv2.TERM_CRITERIA_EPS, 30, 0.001)

            # if corners are found in both images, refine and add data
            if (ret_l and ret_r and ret_rgb):
                self.objpoints.append(self.objp)
                rt = cv2.cornerSubPix(img_l, corners_l, (5, 5),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                rt = cv2.cornerSubPix(img_r, corners_r, (5, 5),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                rt = cv2.cornerSubPix(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), corners_rgb, (5, 5),
                                      (-1, -1), self.criteria)
                self.imgpoints_rgb.append(corners_rgb)

                self.calib_successes_lr.append(polygon_from_image_name(image_left))
                print("\t[OK]. Took %i seconds." % (round(time.time() - start_time, 2)))
            else:
                print("\t[ERROR] - Corners not detected. Took %i seconds." % (round(time.time() - start_time, 2)))

            self.lr_shape = img_r.shape[::-1]
            self.rgb_shape = img_rgb.shape[:2][::-1]
        print(str(len(self.objpoints)) + " of " + str(len(images_left)) +
              " images being used for calibration")
        # self.ensure_valid_images()

    def ensure_valid_images(self):
        """
        Ensures there is one set of left/right images for each polygon. If not, raises an raises an
        AssertionError with instructions on re-running calibration for the invalid polygons.
        """
        expected_polygons = len(setPolygonCoordinates(1000,600)) # inseted values are placeholders
        unique_calib_successes = set(self.calib_successes_lr)
        if len(unique_calib_successes) != expected_polygons:
            valid = set(np.arange(0,expected_polygons))
            missing = valid - unique_calib_successes
            arg_value = ' '.join(map(str, missing))
            raise AssertionError("Missing valid image sets for %i polygons. Re-run calibration with the\n'-p %s' argument to re-capture images for these polygons." % (len(missing), arg_value))
        else:
            return True

    def stereo_calibrate(self):
        """Calibrate camera and construct Homography."""
        # init camera calibrations
        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, self.lr_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, self.lr_shape, None, None)
        rt, self.M3, self.d3, self.r3, self.t3 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_rgb, self.lr_shape, None, None)

        # config
        flags = 0
        #flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        #flags |= cv2.CALIB_RATIONAL_MODEL
        #flags |= cv2.CALIB_FIX_K1
        #flags |= cv2.CALIB_FIX_K2
        #flags |= cv2.CALIB_FIX_K3
        #flags |= cv2.CALIB_FIX_K4
        #flags |= cv2.CALIB_FIX_K5
        #flags |= cv2.CALIB_FIX_K6
        #flags |= cv2.CALIB_FIX_INTRINSIC

        stereocalib_criteria = (cv2.TERM_CRITERIA_COUNT +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # stereo calibration procedure (init)
        ret, self.M1, self.d1, self.M2, self.d2, R_lr, T_lr, E_lr, F_lr = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.lr_shape,
            criteria=stereocalib_criteria, flags=flags)

        # stereo calibration procedure (L-RGB)
        ret, self.M1_rgb, self.d1_rgb, self.M3, self.d3, R_l_rgb, T_l_rgb, E_l_rgb, F_l_rgb = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_rgb,
            self.M1, self.d1, self.M3, self.d3, self.lr_shape,  # image size can be any in case of CALIB_FIX_INTRINSIC
            criteria=stereocalib_criteria, flags=flags)
        print("calibration error (RGB): " + str(ret))
        print("calibration R (L-RGB): \n" + str(R_l_rgb))
        print("calibration T (L-RGB): \n" + str(T_l_rgb))
        print("calibration E (L-RGB): \n" + str(E_l_rgb))
        print("calibration F (L-RGB): \n" + str(F_l_rgb))

        # stereo calibration procedure (R-RGB)
        ret, self.M2_rgb, self.d2_rgb, self.M3, self.d3, R_r_rgb, T_r_rgb, E_r_rgb, F_r_rgb = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_r, self.imgpoints_rgb,
            self.M2, self.d2, self.M3, self.d3, self.lr_shape,
            criteria=stereocalib_criteria, flags=flags)
        print("calibration error (RGB): " + str(ret))
        print("calibration R (R-RGB): \n" + str(R_r_rgb))
        print("calibration T (R-RGB): \n" + str(T_r_rgb))
        print("calibration E (R-RGB): \n" + str(E_r_rgb))
        print("calibration F (R-RGB): \n" + str(F_r_rgb))

        # stereo calibration procedure (init)
        ret, self.M1, self.d1, self.M2, self.d2, R_lr, T_lr, E_lr, F_lr = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            self.M1, self.d1, self.M2, self.d2, self.lr_shape,
            criteria=stereocalib_criteria, flags=flags)
        print("calibration error (LR): " + str(ret))
        print("calibration R (LR): \n" + str(R_lr))
        print("calibration T (LR): \n" + str(T_lr))
        print("calibration E (LR): \n" + str(E_lr))
        print("calibration F (LR): \n" + str(F_lr))

        assert ret < 1.0, "[ERROR] Calibration RMS error < 1.0 (%i). Re-try image capture." % (ret)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.M1, self.d1, self.M2, self.d1, self.lr_shape, R_lr, T_lr)

        np.savez("intrinsics", M1=self.M1, D1=self.d1, M2=self.M2, D2=self.d2, M3=self.M3, D3=self.d3)

        np.savez("extrinsics_lr", R=R_lr, T=T_lr, E=E_lr, F=F_lr,
                R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2)
        np.savez("extrinsics_l_rgb", R=R_l_rgb, T=T_l_rgb, E=E_l_rgb, F=F_l_rgb)
        np.savez("extrinsics_r_rgb", R=R_r_rgb, T=T_r_rgb, E=E_r_rgb, F=F_r_rgb)

        # Error can be large because the rgb image is not synchronized
        # assert ret < 1.0, "[ERROR] Calibration RMS error < 1.0 (%i). Re-try image capture." % (ret)

        print("[OK] Calibration successful w/ RMS error=" + str(ret))

        # construct Homography
        plane_depth = 40000000.0  # arbitrary plane depth 
        #TODO: Need to understand effect of plane_depth. Why does this improve some boards' cals?
        n = np.array([[0.0], [0.0], [-1.0]])
        d_inv = 1.0 / plane_depth
        H = (R_lr - d_inv * np.dot(T_lr, n.transpose()))
        self.H = np.dot(self.M2, np.dot(H, np.linalg.inv(self.M1)))
        self.H /= self.H[2, 2]
        # rectify Homography for right camera
        disparity = (self.M1[0, 0] * T_lr[0] / plane_depth)
        self.H[0, 2] -= disparity
        self.H = self.H.astype(np.float32)
        print("Rectifying Homography...")
        print(self.H)

    def show_rectified_images(self, dataset_dir, calibration_file):
        images_left = glob.glob(dataset_dir + '/left/*.png')
        images_right = glob.glob(dataset_dir + '/right/*.png')
        images_left.sort()
        images_right.sort()

        assert len(images_left) != 0, "ERROR: Images not read correctly"
        assert len(images_right) != 0, "ERROR: Images not read correctly"

        H = np.fromfile(calibration_file, dtype=np.float32).reshape((3, 3))

        print("Using Homography from file, with values: ")
        print(H)

        H = np.linalg.inv(H)
        image_data_pairs = []
        for image_left, image_right in zip(images_left, images_right):
            # read images
            img_l = cv2.imread(image_left, 0)
            img_r = cv2.imread(image_right, 0)

            # warp right image
            img_r = cv2.warpPerspective(img_r, H, img_r.shape[::-1],
                                        cv2.INTER_LINEAR +
                                        cv2.WARP_FILL_OUTLIERS +
                                        cv2.WARP_INVERSE_MAP)

            image_data_pairs.append((img_l, img_r))

        # compute metrics
        imgpoints_r = []
        imgpoints_l = []
        for image_data_pair in image_data_pairs:
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_l, corners_l = cv2.findChessboardCorners(image_data_pair[0],
                                                         (9, 6), flags)
            ret_r, corners_r = cv2.findChessboardCorners(image_data_pair[1],
                                                         (9, 6), flags)

            # termination criteria
            self.criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                             cv2.TERM_CRITERIA_EPS, 30, 0.001)

            # if corners are found in both images, refine and add data
            if ret_l and ret_r:
                rt = cv2.cornerSubPix(image_data_pair[0], corners_l, (5, 5),
                                      (-1, -1), self.criteria)
                rt = cv2.cornerSubPix(image_data_pair[1], corners_r, (5, 5),
                                      (-1, -1), self.criteria)
                imgpoints_l.extend(corners_l)
                imgpoints_r.extend(corners_r)

        epi_error_sum = 0
        for l_pt, r_pt in zip(imgpoints_l, imgpoints_r):
            epi_error_sum += abs(l_pt[0][1] - r_pt[0][1])

        print("Average Epipolar Error: " + str(epi_error_sum / len(imgpoints_r)))

        for image_data_pair in image_data_pairs:
            img_concat = cv2.hconcat([image_data_pair[0], image_data_pair[1]])
            img_concat = cv2.cvtColor(img_concat, cv2.COLOR_GRAY2RGB)

            # draw epipolar lines for debug purposes
            line_row = 0
            while line_row < img_concat.shape[0]:
                cv2.line(img_concat,
                         (0, line_row), (img_concat.shape[1], line_row),
                         (0, 255, 0), 1)
                line_row += 30

            # show image
            print("Displaying Stereo Pair for visual inspection. Press the [ESC] key to exit.")
            while(1):
                cv2.imshow('Stereo Pair', img_concat)
                k = cv2.waitKey(33)
                if k == 32:    # Esc key to stop
                    break
                elif k == 27:
                    os._exit(0)
                    # raise SystemExit()
                elif k == -1:  # normally -1 returned,so don't print it
                    continue
