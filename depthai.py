#!/usr/bin/env python3

import sys
from time import time
from time import sleep
import argparse
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import cv2
import os
import subprocess
import platform
from pathlib import Path

import depthai

import consts.resource_paths
from depthai_helpers import utils

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    WARNING = '\033[1;5;31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def parse_args():
    epilog_text = '''
    Displays video streams captured by DepthAI.

    Example usage:

    ## Show the depth stream:
    python3 test.py -s depth_sipp,12
    ## Show the depth stream and NN output:
    python3 test.py -s metaout previewout,12 depth_sipp,12
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-co", "--config_overwrite", default=None,
                        type=str, required=False,
                        help="JSON-formatted pipeline config object. This will be override defaults used in this script.")
    parser.add_argument("-brd", "--board", default=None, type=str,
                        help="BW1097, BW1098OBC - Board type from resources/boards/ (not case-sensitive). "
                            "Or path to a custom .json board config. Mutually exclusive with [-fv -rfv -b -r -w]")
    parser.add_argument("-fv", "--field-of-view", default=None, type=float,
                        help="Horizontal field of view (HFOV) for the stereo cameras in [deg]. Default: 71.86deg.")
    parser.add_argument("-rfv", "--rgb-field-of-view", default=None, type=float,
                        help="Horizontal field of view (HFOV) for the RGB camera in [deg]. Default: 68.7938deg.")
    parser.add_argument("-b", "--baseline", default=None, type=float,
                        help="Left/Right camera baseline in [cm]. Default: 9.0cm.")
    parser.add_argument("-r", "--rgb-baseline", default=None, type=float,
                        help="Distance the RGB camera is from the Left camera. Default: 2.0cm.")
    parser.add_argument("-w", "--no-swap-lr", dest="swap_lr", default=None, action="store_false",
                        help="Do not swap the Left and Right cameras.")
    parser.add_argument("-e", "--store-eeprom", default=False, action='store_true',
                        help="Store the calibration and board_config (fov, baselines, swap-lr) in the EEPROM onboard")
    parser.add_argument("--clear-eeprom", default=False, action='store_true',
                        help="Invalidate the calib and board_config from EEPROM")
    parser.add_argument("-o", "--override-eeprom", default=False, action='store_true',
                        help="Use the calib and board_config from host, ignoring the EEPROM data if programmed")
    parser.add_argument("-dev", "--device-id", default='', type=str,
                        help="USB port number for the device to connect to. Use the word 'list' to show all devices and exit.")
    parser.add_argument("-debug", "--dev_debug", default=None, action='store_true',
                        help="Used by board developers for debugging.")
    parser.add_argument("-fusb2", "--force_usb2", default=None, action='store_true',
                        help="Force usb2 connection")
    parser.add_argument("-cnn", "--cnn_model", default='mobilenet-ssd', type=str, 
                        help="Cnn model to run on DepthAI")
    parser.add_argument("-dd", "--disable_depth", default=False,  action='store_true', 
                        help="Disable depth calculation on CNN models with bounding box output")
    parser.add_argument("-bb", "--draw-bb-depth", default=False,  action='store_true',
                        help="Draw the bounding boxes over the left/right/depth* streams")
    parser.add_argument("-ff", "--full-fov-nn", default=False,  action='store_true',
                        help="Full RGB FOV for NN, not keeping the aspect ratio")
    parser.add_argument("-s", "--streams",  
                        nargs='+',
                        type=stream_type,
                        dest='streams',
                        default=['metaout', 'previewout'],
                        help="Define which streams to enable \
                        Format: stream_name or stream_name,max_fps \
                        Example: -s metaout previewout \
                        Example: -s metaout previewout,10 depth_sipp,10")
    parser.add_argument("-v", "--video", default=None, type=str, required=False, help="Path where to save video stream (existing file will be overwritten)")

    options = parser.parse_args()

    if (options.board is not None) and ((options.field_of_view     is not None)
                                     or (options.rgb_field_of_view is not None)
                                     or (options.baseline          is not None)
                                     or (options.rgb_baseline      is not None)
                                     or (options.swap_lr           is not None)):
        parser.error("[-brd] is mutually exclusive with [-fv -rfv -b -r -w]")

    # Set some defaults after the above check
    if options.field_of_view     is None: options.field_of_view = 71.86
    if options.rgb_field_of_view is None: options.rgb_field_of_view = 68.7938
    if options.baseline          is None: options.baseline = 9.0
    if options.rgb_baseline      is None: options.rgb_baseline = 2.0
    if options.swap_lr           is None: options.swap_lr = True

    return options

def stream_type(option):
    option_list = option.split(',')
    option_args = len(option_list)
    if option_args not in [1,2]:
        print(bcolors.WARNING + option+" format is invalid. See --help" + bcolors.ENDC)
        raise ValueError

    stream_choices=['metaout', 'previewout', 'left', 'right', 'depth_sipp', 'disparity', 'depth_color_h', 'meta_d2h']
    stream_name = option_list[0]
    if stream_name not in stream_choices:
        print(bcolors.WARNING + stream_name +" is not in available stream list: \n" + str(stream_choices) + bcolors.ENDC)
        raise ValueError

    if(option_args == 1):
        stream_dict = {'name': stream_name}
    else:       
        try:
            max_fps = float(option_list[1])
        except:
            print(bcolors.WARNING + "In option: " + str(option) + " " + option_list[1] + " is not a number!" + bcolors.ENDC)

        stream_dict = {'name': stream_name, "max_fps": max_fps}
    return stream_dict

def decode_mobilenet_ssd(nnet_packet):
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate threw them
    for _, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0 or e[0]['label'] > len(labels):
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        detections.append(e)
    return detections

def nn_to_depth_coord(x, y):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

def average_depth_coord(pt1, pt2):
    factor = 1 - config['depth']['padding_factor']
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2

def show_mobilenet_ssd(entries_prev, frame, is_depth=0):
    img_h = frame.shape[0]
    img_w = frame.shape[1]
    global config
    # iterate through pre-saved entries & draw rectangle & text on image:
    for e in entries_prev:
        # the lower confidence threshold - the more we get false positives
        if e[0]['confidence'] > config['depth']['confidence_threshold']:
            if is_depth:
                pt1 = nn_to_depth_coord(e[0]['left'],  e[0]['top'])
                pt2 = nn_to_depth_coord(e[0]['right'], e[0]['bottom'])
                color = (255, 0, 0) # bgr
                avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2)
                cv2.rectangle(frame, avg_pt1, avg_pt2, color)
                color = (255, 255, 255) # bgr
            else:
                pt1 = int(e[0]['left']  * img_w), int(e[0]['top']    * img_h)
                pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)
                color = (0, 0, 255) # bgr

            x1, y1 = pt1

            cv2.rectangle(frame, pt1, pt2, color)
            # Handles case where TensorEntry object label is out if range
            if e[0]['label'] > len(labels):
                print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
            else:
                pt_t1 = x1, y1 + 20
                cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                pt_t2 = x1, y1 + 40
                cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                if config['ai']['calc_dist_to_bb']:
                    pt_t3 = x1, y1 + 60
                    cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    pt_t4 = x1, y1 + 80
                    cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    pt_t5 = x1, y1 + 100
                    cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
    return frame

# Adjust these thresholds
DETECTION_THRESHOLD = 0.50
IOU_THRESHOLD = 0.1

# Tiny yolo anchor box values
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]


    
def sigmoid(x):
    return 1.0 / (1 + np.exp(x * -1.0))


def calculate_overlap(x1, w1, x2, w2):
    box1_coordinate = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    box2_coordinate = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    overlap = box2_coordinate - box1_coordinate
    return overlap


def calculate_iou(box, truth):
    # calculate the iou intersection over union by first calculating the overlapping height and width
    width_overlap = calculate_overlap(box[0], box[2], truth[0], truth[2])
    height_overlap = calculate_overlap(box[1], box[3], truth[1], truth[3])
    # no overlap
    if width_overlap < 0 or height_overlap < 0:
        return 0

    intersection_area = width_overlap * height_overlap
    union_area = box[2] * box[3] + truth[2] * truth[3] - intersection_area
    iou = intersection_area / union_area
    return iou


def apply_nms(boxes):
    # sort the boxes by score in descending order
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    high_iou_objs = dict()
    # compare the iou for each of the detected objects
    for current_object in range(len(sorted_boxes)):
        if current_object in high_iou_objs:
            continue

        truth = sorted_boxes[current_object]
        for next_object in range(current_object + 1, len(sorted_boxes)):
            if next_object in high_iou_objs:
                continue
            box = sorted_boxes[next_object]
            iou = calculate_iou(box, truth)
            if iou >= IOU_THRESHOLD:
                high_iou_objs[next_object] = 1

    # filter and sort detected items
    filtered_result = list()
    for current_object in range(len(sorted_boxes)):
        if current_object not in high_iou_objs:
            filtered_result.append(sorted_boxes[current_object])
    return filtered_result

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"


def post_processing(output, label_list, threshold):

    num_classes = 20
    num_grids = 13
    num_anchor_boxes = 5
    original_results = output.astype(np.float32)   

    # Tiny Yolo V2 uses a 13 x 13 grid with 5 anchor boxes for each grid cell.
    # This specific model was trained with the VOC Pascal data set and is comprised of 20 classes

    original_results = np.reshape(original_results, (5, 25, 13, 13))
    reordered_results = np.transpose(original_results, (2, 3, 0, 1))
    reordered_results = np.reshape(reordered_results, (169, 5, 25))

    # The 125 results need to be re-organized into 5 chunks of 25 values
    # 20 classes + 1 score + 4 coordinates = 25 values
    # 25 values for each of the 5 anchor bounding boxes = 125 values
    #reordered_results = np.zeros((13 * 13, 5, 25))

    index = 0
    #for row in range( num_grids ):
    #    for col in range( num_grids ):
    #        for b_box_voltron in range(125):
    #            b_box = row * num_grids + col
    #            b_box_num = int(b_box_voltron / 25)
    #            b_box_info = b_box_voltron % 25
    #            reordered_results[b_box][b_box_num][b_box_info] = original_results[row][col][b_box_voltron]

    # shapes for the 5 Tiny Yolo v2 bounding boxes
    anchor_boxes = [1.08,1.19, 3.42,4.41, 6.63,11.38, 9.42,5.11, 16.62,10.52]

    boxes = list()
    # iterate through the grids and anchor boxes and filter out all scores which do not exceed the DETECTION_THRESHOLD
    for row in range(num_grids):
        for col in range(num_grids):
            for anchor_box_num in range(num_anchor_boxes):
                box = list()
                class_list = list()
                current_score_total = 0
                # calculate the coordinates for the current anchor box
                box_x = (col + sigmoid(reordered_results[row * 13 + col][anchor_box_num][0])) / 13.0
                box_y = (row + sigmoid(reordered_results[row * 13 + col][anchor_box_num][1])) / 13.0
                box_w = (np.exp(reordered_results[row * 13 + col][anchor_box_num][2]) *
                         anchor_boxes[2 * anchor_box_num]) / 13.0
                box_h = (np.exp(reordered_results[row * 13 + col][anchor_box_num][3]) *
                         anchor_boxes[2 * anchor_box_num + 1]) / 13.0
                
                # find the class with the highest score
                for class_enum in range(num_classes):
                    class_list.append(reordered_results[row * 13 + col][anchor_box_num][5 + class_enum])



                current_score_total = sum(class_list)
                for current_class in range(len(class_list)):
                    class_list[current_class] = class_list[current_class] * 1.0 / current_score_total

                # probability that the current anchor box contains an item
                object_confidence = sigmoid(reordered_results[row * 13 + col][anchor_box_num][4])
                # highest class score detected for the object in the current anchor box
                highest_class_score = max(class_list)
                # index of the class with the highest score
                class_w_highest_score = class_list.index(max(class_list)) + 1
                # the final score for the detected object
                final_object_score = object_confidence * highest_class_score

                box.append(box_x)
                box.append(box_y)
                box.append(box_w)
                box.append(box_h)
                box.append(class_w_highest_score)
                box.append(object_confidence)
                box.append(highest_class_score)
                box.append(final_object_score)

                # filter out all detected objects with a score less than the threshold
                if final_object_score > threshold:
                    boxes.append(box)

    # gets rid of all duplicate boxes using non-maximal suppression
    results = apply_nms(boxes)

    return results


def decode_tiny_yolo(nnet_packet):

    source_image_width = 416
    source_image_height = 416
    scaled_w = 416
    scaled_h = 416

    # print("nnet_packet.entries()" + str(len(nnet_packet.entries())))
    raw_detections = nnet_packet.get_tensor("out")
    raw_detections.dtype = np.float16  # reinterpret cast
    
    filtered_objects = []
    filtered_objects = post_processing(raw_detections, labels, DETECTION_THRESHOLD)


    return filtered_objects


def show_tiny_yolo(results, original_img, is_depth=0):

    image_width = original_img.shape[1]
    image_height = original_img.shape[0]

    label_list = labels

    # calculate the actual box coordinates in relation to the input image
    print('\n Found this many objects in the image: ' + str(len(results)))
    for box in results:
        box_xmin = int((box[0] - box[2] / 2.0) * image_width)
        box_xmax = int((box[0] + box[2] / 2.0) * image_width)
        box_ymin = int((box[1] - box[3] / 2.0) * image_height)
        box_ymax = int((box[1] + box[3] / 2.0) * image_height)
        # ensure the box is not drawn out of the window resolution
        if box_xmin < 0:
            box_xmin = 0
        if box_xmax > image_width:
            box_xmax = image_width
        if box_ymin < 0:
            box_ymin = 0
        if box_ymax > image_height:
            box_ymax = image_height

        print(" - object: " + YELLOW + label_list[box[4]] + NOCOLOR + " is at left: " + str(box_xmin) + " top: " + str(box_ymin) + " right: " + str(box_xmax) + " bottom: " + str(box_ymax))

        # label shape and colorization
        label_text = label_list[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
        label_background_color = (70, 120, 70) # grayish green background for text
        label_text_color = (255, 255, 255)   # white text

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(box_xmin)
        label_top = int(box_ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]

        # set up the colored rectangle background for text
        cv2.rectangle(original_img, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1),
                      label_background_color, -1)
        # set up text
        cv2.putText(original_img, label_text, (int(box_xmin), int(box_ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    label_text_color, 1)
        # set up the rectangle around the object
        cv2.rectangle(original_img, (int(box_xmin), int(box_ymin)), (int(box_xmax), int(box_ymax)), (0, 255, 0), 2)

    return original_img

def decode_age_gender_recognition(nnet_packet):
    detections = []
    for _, e in enumerate(nnet_packet.entries()):
        if e[1]["female"] > 0.8 or e[1]["male"] > 0.8:
            detections.append(e[0]["age"])  
            if e[1]["female"] > e[1]["male"]:
                detections.append("female")
            else:
                detections.append("male")
    return detections

def show_age_gender_recognition(entries_prev, frame):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]
    if len(entries_prev) != 0:
        age = (int)(entries_prev[0]*100)
        cv2.putText(frame, "Age: " + str(age), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        gender = entries_prev[1]
        cv2.putText(frame, "G: " + str(gender), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))
    return frame

def decode_emotion_recognition(nnet_packet):
    detections = []
    for i in range(len(nnet_packet.entries()[0][0])):
        detections.append(nnet_packet.entries()[0][0][i])
    return detections

def show_emotion_recognition(entries_prev, frame):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]
    e_states = {
        0 : "neutral",
        1 : "happy",
        2 : "sad",
        3 : "surprise",
        4 : "anger"
    }
    if len(entries_prev) != 0:
        max_confidence = max(entries_prev)
        if(max_confidence > 0.7):
            emotion = e_states[np.argmax(entries_prev)]
            cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))

    return frame


def decode_landmarks_recognition(nnet_packet):
    landmarks = []
    for i in range(len(nnet_packet.entries()[0][0])):
        landmarks.append(nnet_packet.entries()[0][0][i])
    
    landmarks = list(zip(*[iter(landmarks)]*2))
    return landmarks

def show_landmarks_recognition(entries_prev, frame):
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    if len(entries_prev) != 0:
        for i in entries_prev:
            try:
                x = int(i[0]*img_h)
                y = int(i[1]*img_w)
            except:
                continue
            # # print(x,y)
            cv2.circle(frame, (x,y), 3, (0, 0, 255))

    frame = cv2.resize(frame, (300, 300))

    return frame

global args
try:
    args = vars(parse_args())
except:
    os._exit(2)

 
stream_list = args['streams']

if args['config_overwrite']:
    args['config_overwrite'] = json.loads(args['config_overwrite'])

print("Using Arguments=",args)

if args['force_usb2']:
    print(bcolors.WARNING + "FORCE USB2 MODE" + bcolors.ENDC)
    cmd_file = consts.resource_paths.device_usb2_cmd_fpath
else:
    cmd_file = consts.resource_paths.device_cmd_fpath

if args['dev_debug']:
    cmd_file = ''
    print('depthai will not load cmd file into device.')

calc_dist_to_bb = True
if args['disable_depth']:
    calc_dist_to_bb = False

decode_nn=decode_mobilenet_ssd
show_nn=show_mobilenet_ssd

if args['cnn_model'] == 'age-gender-recognition-retail-0013':
    decode_nn=decode_age_gender_recognition
    show_nn=show_age_gender_recognition
    calc_dist_to_bb=False

if args['cnn_model'] == 'emotions-recognition-retail-0003':
    decode_nn=decode_emotion_recognition
    show_nn=show_emotion_recognition
    calc_dist_to_bb=False

if args['cnn_model'] == 'yolov2-tiny-voc':
    decode_nn=decode_tiny_yolo
    show_nn=show_tiny_yolo
    calc_dist_to_bb=False

if args['cnn_model'] in ['facial-landmarks-35-adas-0002', 'landmarks-regression-retail-0009']:
    decode_nn=decode_landmarks_recognition
    show_nn=show_landmarks_recognition
    calc_dist_to_bb=False

if args['cnn_model']:
    cnn_model_path = consts.resource_paths.nn_resource_path + args['cnn_model']+ "/" + args['cnn_model']
    blob_file = cnn_model_path + ".blob"
    suffix=""
    if calc_dist_to_bb:
        suffix="_depth"
    blob_file_config = cnn_model_path + suffix + ".json"

blob_file_path = Path(blob_file)
blob_file_config_path = Path(blob_file_config)
if not blob_file_path.exists():
    print(bcolors.WARNING + "\nWARNING: NN blob not found in: " + blob_file + bcolors.ENDC)
    os._exit(1)

if not blob_file_config_path.exists():
    print(bcolors.WARNING + "\nWARNING: NN json not found in: " + blob_file_config + bcolors.ENDC)
    os._exit(1)

with open(blob_file_config) as f:
    data = json.load(f)

try:
    labels = data['mappings']['labels']
except:
    print("Labels not found in json!")


print('depthai.__version__ == %s' % depthai.__version__)
print('depthai.__dev_version__ == %s' % depthai.__dev_version__)

if platform.system() == 'Linux':
    ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if(ret != 0):
        print(bcolors.WARNING + "\nWARNING: Usb rules not found" + bcolors.ENDC)
        print(bcolors.RED + "\nSet rules: \n" \
        """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n""" \
        "sudo udevadm control --reload-rules && udevadm trigger \n" \
        "Disconnect/connect usb cable on host! \n"    + bcolors.ENDC)
        os._exit(1)

if not depthai.init_device(cmd_file, args['device_id']):
    print("Error initializing device. Try to reset it.")
    exit(1)


print('Available streams: ' + str(depthai.get_available_steams()))

# Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
config = {
    # Possible streams:
    # ['left', 'right','previewout', 'metaout', 'depth_sipp', 'disparity', 'depth_color_h']
    # If "left" is used, it must be in the first position.
    # To test depth use:
    # 'streams': [{'name': 'depth_sipp', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
    'streams': stream_list,
    'depth':
    {
        'calibration_file': consts.resource_paths.calib_fpath,
        'padding_factor': 0.3,
        'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
        'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number 
    },
    'ai':
    {
        'blob_file': blob_file,
        'blob_file_config': blob_file_config,
        'calc_dist_to_bb': calc_dist_to_bb,
        'keep_aspect_ratio': not args['full_fov_nn'],
    },
    'board_config':
    {
        'swap_left_and_right_cameras': args['swap_lr'], # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
        'left_fov_deg': args['field_of_view'], # Same on 1097 and 1098OBC
        'rgb_fov_deg': args['rgb_field_of_view'],
        'left_to_right_distance_cm': args['baseline'], # Distance between stereo cameras
        'left_to_rgb_distance_cm': args['rgb_baseline'], # Currently unused
        'store_to_eeprom': args['store_eeprom'],
        'clear_eeprom': args['clear_eeprom'],
        'override_eeprom': args['override_eeprom'],
    }
}

if args['board']:
    board_path = Path(args['board'])
    if not board_path.exists():
        board_path = Path(consts.resource_paths.boards_dir_path) / Path(args['board'].upper()).with_suffix('.json')
        if not board_path.exists():
            print('ERROR: Board config not found: {}'.format(board_path))
            os._exit(2)
    with open(board_path) as fp:
        board_config = json.load(fp)
    utils.merge(board_config, config)
if args['config_overwrite'] is not None:
    config = utils.merge(args['config_overwrite'],config)
    print("Merged Pipeline config with overwrite",config)

if 'depth_sipp' in config['streams'] and ('depth_color_h' in config['streams'] or 'depth_mm_h' in config['streams']):
    print('ERROR: depth_sipp is mutually exclusive with depth_color_h')
    exit(2)
    # del config["streams"][config['streams'].index('depth_sipp')]

# Append video stream if video recording was requested and stream is not already specified
video_file = None
if args['video'] is not None:
    
    # open video file
    try:
        video_file = open(args['video'], 'wb')
        if config['streams'].count('video') == 0:
            config['streams'].append('video')
    except IOError:
        print("Error: couldn't open video file for writing. Disabled video output stream")
        if config['streams'].count('video') == 1:
            config['streams'].remove('video')
    

stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in config['streams']]

# create the pipeline, here is the first connection with the device
p = depthai.create_pipeline(config=config)

if p is None:
    print('Pipeline is not created.')
    exit(3)

nn2depth = depthai.get_nn_to_depth_bbox_mapping()


t_start = time()
frame_count = {}
frame_count_prev = {}
for s in stream_names:
    frame_count[s] = 0
    frame_count_prev[s] = 0

entries_prev = []

process_watchdog_timeout=10 #seconds
def reset_process_wd():
    global wd_cutoff
    wd_cutoff=time()+process_watchdog_timeout
    return

reset_process_wd()


while True:
    # retreive data from the device
    # data is stored in packets, there are nnet (Neural NETwork) packets which have additional functions for NNet result interpretation
    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
    
    packets_len = len(nnet_packets) + len(data_packets)
    if packets_len != 0:
        reset_process_wd()
    else:
        cur_time=time()
        if cur_time > wd_cutoff:
            print("process watchdog timeout")
            os._exit(10)

    for _, nnet_packet in enumerate(nnet_packets):
        entries_prev = decode_nn(nnet_packet)

    for packet in data_packets:
        if packet.stream_name not in stream_names:
            continue # skip streams that were automatically added
        packetData = packet.getData()
        if packetData is None:
            print('Invalid packet data!')
            continue
        elif packet.stream_name == 'previewout':
            
            # the format of previewout image is CHW (Chanel, Height, Width), but OpenCV needs HWC, so we
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = packetData[0,:,:]
            data1 = packetData[1,:,:]
            data2 = packetData[2,:,:]
            frame = cv2.merge([data0, data1, data2])

            nn_frame = show_nn(entries_prev, frame)
            cv2.putText(nn_frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.imshow('previewout', nn_frame)
        elif packet.stream_name == 'left' or packet.stream_name == 'right' or packet.stream_name == 'disparity':
            frame_bgr = packetData
            cv2.putText(frame_bgr, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            cv2.putText(frame_bgr, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            if args['draw_bb_depth']:
                show_nn(entries_prev, frame_bgr, is_depth=True)
            cv2.imshow(packet.stream_name, frame_bgr)
        elif packet.stream_name.startswith('depth'):
            frame = packetData

            if len(frame.shape) == 2:
                if frame.dtype == np.uint8: # grayscale
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                else: # uint16
                    frame = (65535 // frame).astype(np.uint8)
                    #colorize depth map, comment out code below to obtain grayscale
                    frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
                    # frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                    cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
                    cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)
            else: # bgr
                cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.putText(frame, "fps: " + str(frame_count_prev[packet.stream_name]), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255)

            if args['draw_bb_depth']:
                show_nn(entries_prev, frame, is_depth=True)
            cv2.imshow(packet.stream_name, frame)

        elif packet.stream_name == 'jpegout':
            jpg = packetData
            mat = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            cv2.imshow('jpegout', mat)

        elif packet.stream_name == 'video':
            videoFrame = packetData
            videoFrame.tofile(video_file)
        
        elif packet.stream_name == 'meta_d2h':
            str_ = packet.getDataAsStr()
            dict_ = json.loads(str_)

            print('meta_d2h Temp',
                ' CSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['css']),
                ' MSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['mss']),
                ' UPA:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa0']),
                ' DSS:' + '{:6.2f}'.format(dict_['sensors']['temperature']['upa1']))            

        frame_count[packet.stream_name] += 1

    t_curr = time()
    if t_start + 1.0 < t_curr:
        t_start = t_curr

        for s in stream_names:
            frame_count_prev[s] = frame_count[s]
            frame_count[s] = 0


    key = cv2.waitKey(1)
    if key == ord('c'):
        depthai.request_jpeg()
    elif key == ord('q'):
        break


del p  # in order to stop the pipeline object should be deleted, otherwise device will continue working. This is required if you are going to add code after the main loop, otherwise you can ommit it.
depthai.deinit_device()

# Close video output file if was opened
if video_file is not None:
    video_file.close()

print('py: DONE.')

