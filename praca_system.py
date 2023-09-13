import sys
import os
import json
from twilio.rest import Client

import numpy as np
import cv2

from pathlib import Path
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN

# Konfiguracja modelu Mask R-CNN
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # Poprawione z NOM_CLASSES na NUM_CLASSES
    DETECTION_MIN_CONFIDENCE = 0.6

# Funkcja do ekstrakcji współrzędnych boxów z klasy samochodu
def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)
    return np.array(car_boxes)

# Ścieżki i konfiguracja
ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Pobranie przetrenowanego modelu, jeśli nie istnieje
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Inicjalizacja modelu
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Ładowanie pliku JSON z zapisanymi współrzędnymi 
json_path = 'coordinates.json'
with open(json_path, 'r') as f:
    json_coordinates = json.load(f)

# Inicjalizacja Twilio
account_sid = 'Twilio SID'
auth_token = 'Twilio Token'
client = Client(account_sid, auth_token)

# Proces detekcji i porównania
video_capture = cv2.VideoCapture(0)  
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Detekcja
    rgb_image = frame[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)
    r = results[0]
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    # Porównanie z danymi JSON
    for detected_box in car_boxes:
        for json_box in json_coordinates['boxes']:
            iou = mrcnn.utils.compute_iou(detected_box, json_box, [1], [1])
            if iou >= 0.8:
                message = client.messages.create(
                    body="Zwolnione miejsce parkingowe!",
                    from_="Twilio Number",
                    to="Your Number"
                )
                print("Wiadomość wysłana.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
