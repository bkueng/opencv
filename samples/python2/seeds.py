#!/usr/bin/env python

'''
This sample demonstrates SEEDS Superpixels segmentation

Usage:
  seeds.py [<video source>]

'''

import cv2

# relative module
import video

# built-in module
import sys


if __name__ == '__main__':
    print __doc__

    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv2.namedWindow('SEEDS')
    cv2.createTrackbar('Iterations', 'SEEDS', 4, 12, nothing)

    seeds = None

    cap = video.create_capture(fn)
    while True:
        flag, img = cap.read()
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        height,width = hsv_img.shape[:2]

        if not seeds:
            seeds = cv2.createSuperpixelSEEDS(width, height)
            seeds.initialize(3, 4, 4)
        num_iterations = cv2.getTrackbarPos('Iterations', 'SEEDS')

        seeds.iterate(hsv_img, num_iterations)

        result = img
        seeds.drawContoursAroundLabels(result)

        cv2.imshow('SEEDS', result)
        ch = cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
