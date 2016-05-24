#!/usr/bin/env python

# example program to decode LCM image

# LCM
import lcm
from bot_core import *

# zlib for de/compression
import zlib

import os

# matrix processing and visualisation
import numpy as np
import matplotlib.pyplot as plt

import cv2

#def disparity_to_points(disp_img):

def image_handle(channel, data):
    global first_image
    global plotobj

    msg = images_t.decode(data)
    #print "time: ",msg.utime
    #print "number of images ", msg.n_images

    for i in range(msg.n_images):
        img = msg.images[i]
        img_type = msg.image_types[i]
        #print "img type", img_type
        #print "w x h, s", img.width, img.height, img.row_stride
        #print "size", img.size
        #print "format", img.pixelformat, (img.pixelformat==image_t.PIXEL_FORMAT_GRAY)
        for mi in range(img.nmetadata):
            md = img.metadata[mi]
            print md.key, ":", md.value

        # process disparity images
        if img_type == images_t.DISPARITY_ZIPPED or img_type == images_t.DISPARITY:
            # decompress
            if img_type == images_t.DISPARITY_ZIPPED:
                raw_compressed_data = img.data
                #print "compressed data size", len(raw_compressed_data)
                raw_decompressed_data = zlib.decompress(raw_compressed_data)
                #print "decompressed data size", len(raw_decompressed_data)
                #print "compression ratio", float(len(raw_compressed_data))/len(raw_decompressed_data)
                raw_data = raw_decompressed_data

            # use raw data directly
            if img_type == images_t.DISPARITY:
                raw_data = img.data

            #print "bytes per pixel", len(raw_data)/(img.width * img.height)

            # create new 16bit image
            # image format is given as PIXEL_FORMAT_GRAY (8bit single channel)
            # should be PIXEL_FORMAT_BE_GRAY16 or PIXEL_FORMAT_LE_GRAY16 because we use single channel, 16bit per pixel
            # byte order is unknown, assume big-endian
            img16 = np.fromstring(raw_data, dtype=np.uint16)
            img16 = np.reshape(img16, (img.width, img.height))

            #print "img dim", img16.shape

            if plot_depth:
                # live update of image in window
                if first_image:
                    # create new window
                    #plotobj = plt.imshow(img16, cmap='gray')
                    plotobj = plt.matshow(img16)
                    plt.show(block=False)
                    first_image = False;
                else:
                    # plot into existing window
                    plotobj.set_array(img16)
                    plt.draw()

            if write_depth:
                #print np.min(img16), np.max(img16)
                img16 = (np.iinfo(np.uint16).max - img16) # invert disparity values
                #print np.min(img16), np.max(img16)
                #print "img16",img16
                #print "img16 min max", np.min(img16), np.max(img16)
                cv2.imwrite(os.path.join(img_path, "depth_"+str(img.utime)+".png"), img16)
                #img8 = ((img16/float(np.iinfo(np.uint16).max)) * np.iinfo(np.uint8).max).astype(np.uint8)
                #cv2.imwrite(os.path.join(img_path, "depth_" + str(img.utime) + ".png"), img8)


# global variables for updating image view
global first_image
global plotobj
global img_path
global plot_depth
global write_depth

# write or plot images
plot_depth = False
write_depth = True

# flag to set properties for initial window (resolution, value range)
first_image = True

Q = np.zeros((4,4))
Q[0,0] = 1
Q[1,1] = 1
Q[0,3] = -512
Q[1,3] = -512
Q[2,3] = 6.5 # mm
Q[3,2] = -1/0.07
Q[3,3] = -1/0.07

print Q

img_path = "video/"

lc = lcm.LCM()
subs = lc.subscribe("CAMERA", image_handle)

if write_depth:
    if not os.path.exists(img_path):
        os.makedirs(img_path)

try:
    while True:
        lc.handle()
except KeyboardInterrupt:
    pass