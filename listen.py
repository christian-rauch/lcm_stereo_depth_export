#!/usr/bin/env python

# example program to decode LCM image

# LCM
import lcm
from bot_core import *

# zlib for de/compression
import zlib
# jpeg de/compression
from cStringIO import StringIO
from PIL import Image

import os
import sys

# matrix processing and visualisation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cv2

def disparity_to_points(disp_img):
    # disparity image as signed integer 16bit
    points = cv2.reprojectImageTo3D(disparity=disp_img.astype(np.int16), Q=Q_transf_mat)
    return points

def image_handle(channel, data):
    global first_image
    global plotobj
    global plot3d_obj
    global plot3d_ax

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
                    first_image = False
                else:
                    # plot into existing window
                    plotobj.set_array(img16)
                    plt.draw()

            if plot_points:
                p = disparity_to_points(img16)
                if first_image:
                    fig = plt.figure()
                    plot3d_ax = fig.add_subplot(111, projection='3d')
                    plot3d_ax.scatter(p[:,1], p[:,2], p[:,3])
                    plt.show(block=False)
                    first_image = False
                else:
                    plot3d_ax.clear()
                    plot3d_ax.scatter(p[:, 1], p[:, 2], p[:, 3])
                    plt.draw()

            if write_depth:
                if export_distance:
                    # Z = (f*b)/d
                    img16_depth = (556.183166504 * 0.07) / (img16 * (1.0/16.0)) # distance in meter
                    img16_depth = np.around(img16_depth * 1000) # distance in mm
                    #print "img16 min max", np.min(img16_depth), np.max(img16_depth[~np.isinf(img16_depth)]), img16_depth.dtype
                    img16_depth = img16_depth.astype(dtype=np.uint16)
                    # set disparity 0 to maximum distance
                    #img16_depth[img16 == 0] = np.iinfo(np.uint16).max
                    img16 = img16_depth
                    #print "img16 min max", np.min(img16), np.max(img16[~np.isinf(img16)]), img16.dtype
                    img_type_str = "depth"
                else:
                    img_type_str = "disparity"
                #print np.min(img16), np.max(img16)
                #img16 = (np.iinfo(np.uint16).max - img16) # invert disparity values
                #print np.min(img16), np.max(img16)
                #print "img16",img16
                #print "img16 min max", np.min(img16), np.max(img16)
                #print "img16 min max", np.min(img16), np.max(img16[~np.isinf(img16)]), img16.dtype
                #img16 = img16.astype(np.uint16)
                #print "img16 min max", np.min(img16), np.max(img16[~np.isinf(img16)]), img16.dtype
                cv2.imwrite(os.path.join(img_path, img_type_str+"_"+str(img.utime)+".png"), img16)
                #img8 = ((img16/float(np.iinfo(np.uint16).max)) * np.iinfo(np.uint8).max).astype(np.uint8)
                #cv2.imwrite(os.path.join(img_path, "depth_" + str(img.utime) + ".png"), img8)

        if img_type == images_t.LEFT: # or img_type == images_t.RIGHT:
            datafile = StringIO(img.data)
            colourdata = Image.open(datafile)
            colourdata.save(os.path.join(img_path, "colour_" + str(img.utime) + ".png"))

if __name__ == "__main__":
    # global variables for updating image view
    global first_image
    global plotobj
    global img_path
    global plot_depth
    global write_depth
    global plot3d_obj
    global plot3d_ax

    # write or plot images
    plot_depth = False
    write_depth = True
    plot_points = False

    export_distance = True # distance or disparity images, True will convert LCM disparity to distance

    # flag to set properties for initial window (resolution, value range)
    first_image = True

    Q_transf_mat = np.zeros((4, 4))
    Q_transf_mat[0, 0] = 1
    Q_transf_mat[1, 1] = 1
    Q_transf_mat[0, 3] = -512
    Q_transf_mat[1, 3] = -512
    Q_transf_mat[2, 3] = 556.18 # pxl
    Q_transf_mat[3, 2] = -1 / -0.07 # B=7cm
    Q_transf_mat[3, 3] = 0

    print Q_transf_mat

    img_path = "video/"

    if len(sys.argv) > 1:
        print "reading from log: ", sys.argv[1]
        lc = lcm.LCM("file://" + sys.argv[1])
    else:
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