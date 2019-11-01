#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import time

import cv2 as cv

from airtest.core.api import connect_device, device, shell
from airtest.core.error import AdbShellError


class Phone(object):
    def __init__(self, serial_no, cap_method='javacap', touch_method='adb'):
        self.serial_no = str(serial_no)
        self.cap_method = str(cap_method)
        self.touch_method = str(touch_method)

        # 连接设备
        device_url = 'android:///%s?cap_method=%s&touch_method=%s' % (
        self.serial_no, self.cap_method, self.touch_method)
        _device = connect_device(device_url)
        self.device = _device
        print("device connect done!")

        self._rx1 = None
        self._tx1 = None
        self._rx2 = None
        self._tx2 = None

    def shell(self, *args, **kwargs):
        result = self.device.shell(*args, **kwargs)
        return result

    def save_image(self, image, filename):
        cv.imwrite(filename, image)
        return filename

    def snapshot(self, save=False):
        image = self.device.snapshot()
        if save:
            self.save_image(image, filename=f'../dataset/image/snapshot_{str(int(time.time()))}.jpg')
        return image


if __name__ == '__main__':
    # serial_no = '139.159.250.28:8126'
    # serial_no = '139.159.250.28:8129'
    # serial_no = '139.159.250.28:8132'
    serial_no = '139.159.250.28:8135'
    phone = Phone(serial_no=serial_no)
    phone.snapshot(save=True)
