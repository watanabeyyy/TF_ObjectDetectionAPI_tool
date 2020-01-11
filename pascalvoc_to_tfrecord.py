"""create TFrecord from PascalVoc-style .xml file
"""
import os
import io
import glob
import argparse
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random

from tqdm import tqdm
from PIL import Image
from object_detection.utils import dataset_util

''' INSTRUCTION
This script performs the following:
(1) Divides dataset into training and evaluation (90:10)
(2) Shuffles the dataset before converting it into TFrecords


Expected directories structure:
VOC_dataset 
   -JPEGImages
   -Annotations
    convert_to_tfrecord.py (this script)

To run this script:
$ python convert_to_tfrecord.py 

END INSTRUCTION '''


def create_example(xml_file, input_img_dir):
    # process the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find('filename').text
    file_name = image_name.encode('utf8')
    size = root.find('size')
    width = int(size[0].text)
    height = int(size[1].text)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for member in root.findall('object'):
        name = member[0].text
        classes_text.append(name.encode('utf8'))
        xmin.append(float(member[4][0].text) / width)
        ymin.append(float(member[4][1].text) / height)
        xmax.append(float(member[4][2].text) / width)
        ymax.append(float(member[4][3].text) / height)
        difficult_obj.append(0)
        # For multiple classes, change the code block to read
        # the classes from the Annotations xml as following:
        # if name == "ball":
        #     classes.append(0)
        if name == "white":
            classes.append(1)
        if name == "black":
            classes.append(2)
        truncated.append(0)
        poses.append('Unspecified'.encode('utf8'))

    full_path = os.path.join(input_img_dir, '{}'.format(image_name))
    with open(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    # Create TFRecord
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-i', '--input_img_dir',
                        default="",
                        help="")
    parser.add_argument('-x', '--input_xml_dir',
                        default="",
                        help="")
    args = parser.parse_args()

    input_img_dir = args.input_img_dir
    input_xml_dir = args.input_xml_dir
    xml_path = os.path.join(input_xml_dir, "*.xml")
    # filename_list = tf.train.match_filenames_once(input_xml_dir)
    # init = (tf.global_variables_initializer(),
    #         tf.local_variables_initializer())
    # sess = tf.Session()
    # sess.run(init)
    # list = sess.run(filename_list)
    list = glob.glob(xml_path)
    random.shuffle(list)  # shuffle files list
    for i, xml_file in enumerate(tqdm(list)):
        # 五桁の連番で命名
        filename = "{:05d}".format(i) + ".tfrecord"
        if (i % 10) != 0:
            writer = tf.io.TFRecordWriter(
                "data/train/"+filename)
        else:
            writer = tf.io.TFRecordWriter(
                "data/val/"+filename)
        example = create_example(xml_file, input_img_dir)
        writer.write(example.SerializeToString())
        writer.close()
