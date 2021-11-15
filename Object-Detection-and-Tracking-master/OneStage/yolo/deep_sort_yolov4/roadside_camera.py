import glob
import os
import sys
from typing import Counter
try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Carla not found")
    pass
import carla

import random
import time
import numpy as np
import cv2
import torch

IM_WIDTH = 1024
IM_HEIGHT = 1024

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
c = 1
def process_img(image, str):
    global c
    i = np.array(image.raw_data)
    # rgba, so 4
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    # cv2.imshow(str, i3)
    # cv2.waitKey(1)
    if image.frame % 30 == 0:
        #image.save_to_disk('_out/%04d.jpg' % image.frame)
        print(f"saving capture #{c}")
        image.save_to_disk(f'_out/{c}.jpg')
        c += 1
    # normalize data btw 0-1
    return i3/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000) # https://carla.readthedocs.io/en/0.9.11/core_world/#the-client
    client.set_timeout(3.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library() # https://carla.readthedocs.io/en/0.9.11/core_actors/#blueprints

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("sensor_tick", "5.0")
    
    sp = carla.Location(-98.6, -12.6, 15.8)
    cam_rotation = carla.Rotation(-30,40,0)
    cam_transform = carla.Transform(sp,cam_rotation)

    ego_cam = world.spawn_actor(cam_bp,cam_transform)

    actor_list.append(ego_cam)

    ego_cam.listen(lambda data: process_img(data, "view"))

    time.sleep(300)
    

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')