import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn
from PIL import Image
import time
import numpy as np
import os

# Inicjalizacja modeli
mask_rcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
mask_rcnn_model.eval()
faster_rcnn_model.eval()

# Listy do przechowywania czasu detekcji i dokładności
mask_rcnn_times = []
faster_rcnn_times = []

# Folder z obrazami do testowania
image_folder = "image_test_folder"

# Pętla przez 100 obrazów
for idx, image_name in enumerate(os.listdir(image_folder)[:100]):
    image_path = os.path.join(image_folder, image_name)
    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    
    # Testowanie Mask R-CNN
    start_time = time.time()
    with torch.no_grad():
        mask_rcnn_pred = mask_rcnn_model([img])
    elapsed_time = time.time() - start_time
    mask_rcnn_times.append(elapsed_time)
    
    # Testowanie Faster R-CNN
    start_time = time.time()
    with torch.no_grad():
        faster_rcnn_pred = faster_rcnn_model([img])
    elapsed_time = time.time() - start_time
    faster_rcnn_times.append(elapsed_time)

# Obliczenie statystyk
def calculate_statistics(time_list):
    return {
        'count': len(time_list),
        'mean': np.mean(time_list),
        'std': np.std(time_list),
        'min': np.min(time_list),
        '25%': np.percentile(time_list, 25),
        '50%': np.median(time_list),
        '75%': np.percentile(time_list, 75),
        'max': np.max(time_list)
    }

stats_mask_rcnn = calculate_statistics(mask_rcnn_times)
stats_faster_rcnn = calculate_statistics(faster_rcnn_times)

# Wyświetlenie statystyk
print("Statystyki dla Mask R-CNN:", stats_mask_rcnn)
print("Statystyki dla Faster R-CNN:", stats_faster_rcnn)
