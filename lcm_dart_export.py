#!/usr/bin/env python
import lcm
import sys
import zlib

import csv

import numpy as np
import cv2

import os

from bot_core import images_t, robot_state_t


class Export:
    def __init__(self):
        self.wrote_names = False

    def last_joint_state(self, msg):
        if not self.wrote_names:
            self.joint_names = msg.joint_name
        self.joint_values = msg.joint_position

    def depth_img_to_file(self, msg):
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
                cv2.imwrite(os.path.join(img_path, img_type_str+"_"+str(img.utime)+".png"), img16)
                #print "write", self.joint_values
                #print "write",[str(img.utime)]+list(self.joint_values)
                if not self.wrote_names:
                    namewriter = csv.writer(joint_name_file, delimiter=',')
                    namewriter.writerow(list(self.joint_names))
                    #csvwriter.writerow(["#time"] + list(self.joint_names))
                    joint_name_file.close()
                    self.wrote_names = True
                timewriter.writerow([str(img.utime)])
                #csvwriter.writerow([str(img.utime)]+list(self.joint_values))
                csvwriter.writerow(list(self.joint_values))


if __name__ == "__main__":
    export_distance = True
    log = lcm.EventLog(sys.argv[1], "r")

    export_folder = "export"
    img_folder = "video"
    joint_folder = "joints"

    img_path = os.path.join(export_folder, img_folder)
    joint_path = os.path.join(export_folder, joint_folder)

    exporter = Export()

    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if not os.path.exists(joint_path):
        os.makedirs(joint_path)

    joint_file = open(os.path.join(joint_path, "joints.csv"), 'w')
    csvwriter = csv.writer(joint_file, delimiter=',')

    joint_name_file = open(os.path.join(joint_path, "joint_namess.csv"), 'w')

    timestamp_file = open(os.path.join(joint_path, "timestamps.csv"), 'w')
    timewriter = csv.writer(timestamp_file, delimiter=',')

    for event in log:
        if event.channel == "CAMERA":
            msg = images_t.decode(event.data)
            exporter.depth_img_to_file(msg)
        if event.channel == "EST_ROBOT_STATE":
            msg = robot_state_t.decode(event.data)
            exporter.last_joint_state(msg)

    joint_file.close()
    timestamp_file.close()