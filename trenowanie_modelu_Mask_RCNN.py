import os
import json
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Model

# Wczytanie etykiet z pliku JSON
with open("labels.json", "r") as f:
    labels_data = json.load(f)

# Wczytanie obrazów i etykiet
image_folder = "image_data"

image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
images = [cv2.imread(path) for path in image_paths]
images = [cv2.resize(img, (128, 128)) for img in images]

y_class = [d['class'] for d in labels_data]
y_bbox = [d['bbox'] for d in labels_data]
y_mask = [d['mask'] for d in labels_data]

x_data = np.array(images)
y_class_data = np.array(y_class)
y_bbox_data = np.array(y_bbox)
y_mask_data = np.array(y_mask)

# Podział danych
split_index = int(0.8 * len(x_data))
x_train, x_val = x_data[:split_index], x_data[split_index:]
y_class_train, y_class_val = y_class_data[:split_index], y_class_data[split_index:]
y_bbox_train, y_bbox_val = y_bbox_data[:split_index], y_bbox_data[split_index:]
y_mask_train, y_mask_val = y_mask_data[:split_index], y_mask_data[split_index:]

# Konfiguracja modelu
input_shape = (128, 128, 3)
n_classes = 2  # liczba klas

# Tworzenie modelu

# Podstawowa sieć
input_img = Input(shape=input_shape, name='image_input')

# Warstwy konwolucyjne i pooling
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Wyjście klasyfikacji
class_output = Dense(n_classes, activation='softmax', name='class_output')(encoded)

# Wyjście bounding box
bbox_output = Dense(4, activation='linear', name='bbox_output')(encoded)

# Wyjście maski
mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(encoded)

# Cały model
model = Model(inputs=input_img, outputs=[class_output, bbox_output, mask_output])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy',
                    'bbox_output': 'mean_squared_error',
                    'mask_output': 'binary_crossentropy'},
              metrics={'class_output': 'accuracy',
                       'bbox_output': 'mse',
                       'mask_output': 'accuracy'})

# Trening modelu
model.fit(x_train, {'class_output': y_class_train, 'bbox_output': y_bbox_train, 'mask_output': y_mask_train},
          validation_data=(x_val, {'class_output': y_class_val, 'bbox_output': y_bbox_val, 'mask_output': y_mask_val}),
          epochs=10,
          batch_size=16)

# Ocena modelu na zestawie walidacyjnym
wyniki = model.evaluate(x_val, {'class_output': y_class_val, 'bbox_output': y_bbox_val, 'mask_output': y_mask_val})
