from ultralytics import YOLO
from PIL import Image
import os
import cv2

#model=YOLO("yolov8l.pt")
model=YOLO("train_100_gcolab\weights\last.pt") 




img='197113716.jpg'

results = model.predict(img, stream=True)                 # run prediction on img
print(results.xyxy[0]) 
# for result in results:                                         # iterate results
#     boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
#     for box in boxes:                                          # iterate boxes
#         r = box.xyxy[0].astype(int)                            # get corner points as int
#         print(r)                                               # print boxes
#         #cv2.rectangle(img, r[:2], r[2:], (255, 255, 255), 2)   # draw boxes on img