from ultralytics import YOLO
#from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import os
import cv2

model=YOLO("yolov8l.pt")
#model=YOLO("train_100_gcolab\weights\last.pt")

model.predict(source="0",show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=2)
#results=model.predict(source='0',show=True)
#print(results)