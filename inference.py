import argparse
import cv2
import os
import numpy as np
import glob
import time
from utils.detection import DetectionModel_PB, DetectionModel_TFLite, inference_tta
from utils.detection_tools import non_max_suppression, overlay_result

parser = argparse.ArgumentParser(description="")
parser.add_argument('-i', '--input_dir',
                    default="",
                    help="")
parser.add_argument('-m', '--model',
                    default="",
                    help=".pb or .tflite")
args = parser.parse_args()

input_dir = args.input_dir
model = args.model
if ".pb" in model:
    detection_model = DetectionModel_PB(model)
elif ".tflite" in model:
    detection_model = DetectionModel_TFLite(model)
else:
    print("invalid model file")

img_paths = glob.glob(os.path.join(input_dir, "*.png"))
print("img num = ", len(img_paths))
for path in img_paths:
    img = cv2.imread(path)
    a = time.time()
    objects = detection_model.inference(img)
    # objects = inference_tta(detection_model, img)
    a = time.time() - a
    print("inference_time = ", a)
    objects = non_max_suppression(objects, 0.5, img)
    detected_img = overlay_result(objects, img)
    cv2.imshow("", detected_img)
    cv2.waitKey(0)
