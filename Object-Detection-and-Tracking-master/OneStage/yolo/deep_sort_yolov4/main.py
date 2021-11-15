#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import random
import carla
import threading

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="./test_video/TownCentreXVID.avi")
ap.add_argument("-c", "--class", help="name of class", default="person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")
_HOST_ = '127.0.0.1'
_PORT_ = 2000
IM_WIDTH = 640
IM_HEIGHT = 480
random.seed(0)

sensor1_buf = []
frame_read = 0


def process_img(image, str):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    sensor1_buf.append(np.ascontiguousarray(i3, dtype=np.uint8))
    cv2.imshow(str, i3)
    cv2.waitKey(1)


def Carla():
    client = carla.Client(_HOST_, 2000)  # https://carla.readthedocs.io/en/0.9.11/core_world/#the-client
    client.set_timeout(200.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle = blueprint_library.filter('vehicle')[1]  # hard code car
    # vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    sensor = blueprint_library.find('sensor.camera.rgb')
    # randomly picking a point, seed = 0 here
    spawn_points = random.choice(world.get_map().get_spawn_points())
    # change the dimensions of the image
    sensor.set_attribute('image_size_x', f'{IM_WIDTH}')
    sensor.set_attribute('image_size_y', f'{IM_HEIGHT}')
    sensor.set_attribute('fov', '110')
    actor_vehicle = world.spawn_actor(blueprint=vehicle, transform=spawn_points)
    # print(spawn_points.get_matrix())
    spawn_point_sensor = carla.Transform(carla.Location(x=105, y=44, z=8), carla.Rotation(pitch=-30, yaw=0, roll=0))
    actor_sensor = world.spawn_actor(blueprint=sensor, transform=spawn_point_sensor)
    # walker = blueprint_library.find('walker.pedestrian.0005')
    # print("~~~~~~~")
    # walker_point = carla.Transform(carla.Location(x=109, y=53, z=3))
    # walker = world.spawn_actor(blueprint=walker, transform=walker_point)

    # controller_walker_bp = blueprint_library.find('controller.ai.walker')
    # controller_walker = world.spawn_actor(blueprint=controller_walker_bp,
    #                                       transform=carla.Transform(carla.Location(x=0, y=0, z=0)), attach_to=walker)
    # controller_walker.start()
    # set walk to random point
    # controller_walker.go_to_location(carla.Location(x=109, y=53, z=1))
    # random max speed
    # controller_walker.set_max_speed(0.2)  # max speed between 1 and 2 (default is 1.4 m/s)
    actor_sensor.listen(lambda data: process_img(data, "sensor2"))
    actor_vehicle.set_autopilot(True)
    while True:
        continue


# list = [[] for _ in range(100)]
def yolo():
    main(YOLO())


def main(yolo):
    print("started")
    FRAMEDETECT = 0
    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    counter = []
    # deep_sort
    model_filename = 'model_data/yolo4_weight.h5'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    # video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        # w = int(video_capture.get(3))
        # h = int(video_capture.get(4))
        w = 640
        h = 480
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # out = cv2.VideoWriter('./output/output.avi', fourcc, 15, (w, h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        # When input is images, we can use imread to extract numpy ndarray
        # im = cv2.imread("abc.tiff",mode='RGB')

        # ret, frame = video_capture.read()  # frame shape 640*480*3
        # if ret != True:
        #     break
        buflen = len(sensor1_buf)
        if buflen == 0:
            continue

        ########################################################################################
        # todo: For performance issue, we need to control the frames caught by sensor:
        # 0. if you want to check latest frame in buffer, reducing latency as much as possible
        frame = sensor1_buf.pop()
        # 1.
        # frame = sensor1_buf.pop(0) if you want to check each frame
        # 2.
        # if you want to skip by n frames per detection, n=15 here
        # frame = sensor1_buf[FRAMEDETECT]
        # if FRAMEDETECT+15 >= buflen:
        #     FRAMEDETECT = buflen-1
        # else:
        #     FRAMEDETECT += 15
        #######################################################################################
        t1 = time.time()
        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb

        boxs, confidence, class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # print(class_names)
            # print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # print(frame_index)
            list_file.write(str(frame_index) + ',')
            list_file.write(str(track.track_id) + ',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (color), 3)
            b0 = str(bbox[0])  # .split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])  # .split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2] - bbox[0])  # .split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3] - bbox[1])

            list_file.write(str(b0) + ',' + str(b1) + ',' + str(b2) + ',' + str(b3))
            # print(str(track.track_id))
            list_file.write('\n')
            # list_file.write(str(track.track_id)+',')
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, (color), 2)
            if len(class_names) > 0:
                class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)

            i += 1
            # bbox_center_point(x,y)
            center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
            # track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            # center point
            cv2.circle(frame, (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), (color), thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        cv2.putText(frame, "Total Pedestrian Counter: " + str(count), (int(20), int(120)), 0, 5e-3 * 200, (0, 255, 0),
                    2)
        cv2.putText(frame, "Current Pedestrian Counter: " + str(i), (int(20), int(80)), 0, 5e-3 * 200, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f" % (fps), (int(20), int(40)), 0, 5e-3 * 200, (0, 255, 0), 3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0)
        cv2.resizeWindow('YOLO4_Deep_SORT', 640, 480)
        cv2.imshow('YOLO4_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            # out.write(frame)
            frame_index = frame_index + 1

        fps = (fps + (1. / (time.time() - t1))) / 2
        # out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
        print(args["input"][43:57] + ": " + str(count) + " " + str(class_name) + ' Found')

    else:
        print("[No Found]")
    # print("[INFO]: model_image_size = (960, 960)")
    # video_capture.release()
    if writeVideo_flag:
        # out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_to_detect = 0
    yolo_thread = threading.Thread(target=yolo)
    yolo_thread.start()
    time.sleep(25)
    print("loaded yolo")
    carla_thread = threading.Thread(target=Carla)
    carla_thread.start()
    time.sleep(30)
    #todo stop thread when in need

