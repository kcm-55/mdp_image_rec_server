import argparse
import math
import os
import shutil
import time
from pathlib import Path
import uuid
import imagezmq

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import urllib.request
from numpy import random
from model import *

DATA_DIR = Path("./data")
RESULT_DIR = Path("./results")


def del_previous_dir():
    try:
        shutil.rmtree("./data")
        shutil.rmtree("./results")
    except OSError as e:
        pass


def capture():
    model = load_model()
    # print("Capturing image...")
    # filename = str(uuid.uuid4())
    # DATA_DIR.mkdir(parents=True, exist_ok=True)  # make dir
    # img_file_path = str(DATA_DIR / filename) + ".jpg"


    # # Capture with webcam
    # cam_port = 0
    # cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
    # print(cam.isOpened())
    # #cam = cv2.VideoCapture(cam_port)
    # result, image = cam.read()
    # if result:
    #     print("The captured image is saved in: ", img_file_path)
    #     cv2.imwrite(img_file_path, image)
    #     image_id = predict_image(img_file_path, model, "C")
    #     return image_id


    # test for rpi connection
    image_hub = imagezmq.ImageHub()
    while True:
        filename = str(uuid.uuid4())
        DATA_DIR.mkdir(parents=True, exist_ok=True)  # make dir
        img_file_path = str(DATA_DIR / filename) + ".jpg"
        print(img_file_path)

        print("capturing")
        rpi_name, image = image_hub.recv_image()
        cv2.imwrite(img_file_path, image)
        image_id = predict_image(img_file_path, model, "C")
        # cv2.imshow(rpi_name, image)
        # cv2.waitKey(1)
        image_hub.send_reply(str(image_id).encode('utf-8'))


def del_previous_dir():
    try:
        shutil.rmtree("./data")
        shutil.rmtree("./runs")
    except OSError as e:
        pass

# capture()