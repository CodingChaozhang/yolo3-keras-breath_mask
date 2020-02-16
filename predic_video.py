#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/13 21:46
# @Author  : codingchaozhang
from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO,detect_video
from PIL import Image

yolo = YOLO()

while True:

    try:
        detect_video(yolo)
    except:
        print('Open Error! Try again!')
        continue


yolo.close_session()
