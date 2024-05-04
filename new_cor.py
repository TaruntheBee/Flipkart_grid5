import torch
from ultralytics import YOLO
import cv2
from PIL import Image

# Model
model=YOLO("train_100_gcolab\weights\last.pt") 

# Images
img='197113716.jpg'

# Inference
results=model(img)
#results = model(source='0',show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False, save_crop=False, line_width=2)
boxes = results[0].boxes
for box in boxes:
    print(box.xywh)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])