import os
import cv2
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import json

class MaskRCNNConfig(Config):
    # Nazwa konfiguracji
    NAME = "pretrained_model_config"
    # Liczba obrazów na każdą kartę graficzną
    IMAGES_PER_GPU = 1
    # Liczba kart graficznych do używania
    GPU_COUNT = 1
    # Liczba klas (80 klas + 1 dla tła)
    NUM_CLASSES = 1 + 80  
    # Minimalna pewność detekcji, żeby ją zaakceptować
    DETECTION_MIN_CONFIDENCE = 0.6

# Funkcja do filtrowania tylko interesujących nas detekcji (samochody, busy itd.)
def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3, 8, 6]:  # indeksy dla samochodów i busów
            car_boxes.append(box)
    return np.array(car_boxes)

# Ścieżki
ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Pobranie przetrenowanego modelu, jeśli nie istnieje
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Inicjalizacja modelu w trybie wnioskowania (inference)
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Wczytanie przetrenowanego modelu
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Inicjalizacja strumienia wideo
VIDEO_SOURCE = "https://www.youtube.com/watch?v=c38K8IsYnB0" # Zamień na własny link
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Konwersja kolorów z BGR na RGB
    rgb_image = frame[:, :, ::-1]
    # Detekcja obiektów
    results = model.detect([rgb_image], verbose=0)
    r = results[0]

    # Filtrowanie detekcji
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    # Rysowanie prostokątów dookoła detekcji
    for box in car_boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Dodanie współrzędnych do listy
    coordinates_list.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Zapis współrzędnych do pliku JSON
    with open("coordinates.json", "w") as json_file:
        json.dump(coordinates_list, json_file)

    # Wyświetlenie obrazu
    cv2.imshow('Video', frame)

    # Wyjście po naciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zamknięcie okna
video_capture.release()
cv2.destroyAllWindows()
