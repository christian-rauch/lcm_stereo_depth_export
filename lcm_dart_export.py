#!/usr/bin/env python
import lcm
import sys
import zlib
# jpeg de/compression
from cStringIO import StringIO
from PIL import Image
from PIL import ImageFile

import csv

import numpy as np
import cv2

import os

from bot_core import images_t, robot_state_t


class Export:
    def __init__(self):
        self.wrote_names_multisense = False
        self.wrote_names_openni = False

    # store curent joint values, this will always store the most recent values
    # we hope this gets called before any image arrives
    def last_joint_state(self, msg):
        if not (self.wrote_names_multisense or self.wrote_names_openni):
            self.joint_names = msg.joint_name
        self.joint_values = msg.joint_position

    def depth_img_to_file(self, msg, img_path):
        for i in range(msg.n_images):
            img = msg.images[i]
            img_type = msg.image_types[i]
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

                # create new 16bit image
                # image format is given as PIXEL_FORMAT_GRAY (8bit single channel)
                # should be PIXEL_FORMAT_BE_GRAY16 or PIXEL_FORMAT_LE_GRAY16 because we use single channel, 16bit per pixel
                # byte order is unknown, assume big-endian
                img16 = np.fromstring(raw_data, dtype=np.uint16)
                img16 = np.reshape(img16, (img.width, img.height))

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
                cv2.imwrite(os.path.join(img_path, img_type_str+"_"+str(msg.utime)+".png"), img16)

                # write joint values associated wih image
                if not self.wrote_names_multisense:
                    csvwriter_multisense.writerow(list(self.joint_names))
                    # for jn in self.joint_names:
                    #     joint_name_file.write(jn + '\n')
                    # joint_name_file.close()
                    self.wrote_names_multisense = True
                timewriter_multisense.writerow([str(msg.utime)])
                # csvwriter_multisense.writerow([str(msg.utime)]+list(self.joint_values))
                csvwriter_multisense.writerow(list(self.joint_values))

            if img_type == images_t.DEPTH_MM_ZIPPED or img_type == images_t.DEPTH_MM:
                img_type_str = "depth"
                if img_type == images_t.DEPTH_MM_ZIPPED:
                    raw_compressed_data = img.data
                    raw_decompressed_data = zlib.decompress(raw_compressed_data)
                    raw_data = raw_decompressed_data
                if img_type == images_t.DEPTH_MM:
                    raw_data = img.data

                img16 = np.fromstring(raw_data, dtype=np.uint16)
                img16 = np.reshape(img16, (img.height, img.width))

                cv2.imwrite(os.path.join(img_path, img_type_str + "_" + str(msg.utime) + ".png"), img16)

                # write joint values associated wih image
                if not self.wrote_names_openni:
                    csvwriter_openni.writerow(list(self.joint_names))
                    self.wrote_names_openni = True
                timewriter_openni.writerow([str(msg.utime)])
                csvwriter_openni.writerow(list(self.joint_values))

            if img_type == images_t.LEFT:  # or img_type == images_t.RIGHT:
                datafile = StringIO(img.data)
                colourdata = Image.open(datafile)
                colourdata.save(os.path.join(img_path, "colour_" + str(msg.utime) + ".png"))


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # export_distance: True => depth, False => disparity
    export_distance = True
    log = lcm.EventLog(sys.argv[1], "r")

    export_folder = "export"
    img_folder = "video"
    joint_folder = "joints"


    joint_path = os.path.join(export_folder, joint_folder)

    exporter = Export()

    if not os.path.exists(joint_path):
        os.makedirs(joint_path)

    csvwriter_multisense = csv.writer(open(os.path.join(joint_path, "joints_multisense.csv"), 'w'), delimiter=',')
    csvwriter_openni = csv.writer(open(os.path.join(joint_path, "joints_openni.csv"), 'w'), delimiter=',')

    timewriter_multisense = csv.writer(open(os.path.join(joint_path, "timestamps_multisense.csv"), 'w'), delimiter=',')
    timewriter_openni = csv.writer(open(os.path.join(joint_path, "timestamps_openni.csv"), 'w'), delimiter=',')

    for event in log:
        if event.channel == "CAMERA":
            img_path = os.path.join(export_folder, img_folder+"_multisense")
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            msg = images_t.decode(event.data)
            exporter.depth_img_to_file(msg, img_path)

        if event.channel == "OPENNI_FRAME":
            img_path = os.path.join(export_folder, img_folder + "_openni")
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            msg = images_t.decode(event.data)
            exporter.depth_img_to_file(msg, img_path)

        if event.channel == "EST_ROBOT_STATE":
            msg = robot_state_t.decode(event.data)
            exporter.last_joint_state(msg)