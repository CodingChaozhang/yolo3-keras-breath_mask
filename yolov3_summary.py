#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 20:58
# @Author  : codingchaozhang
from nets.yolo3 import yolo_body
from keras.layers import Input

Inputs = Input([416,416,3])
model = yolo_body(Inputs,3,20)
model.save("yolov3.h5")
model.summary()