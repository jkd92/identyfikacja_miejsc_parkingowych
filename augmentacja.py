from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# Inicjalizacja generatora augmentacji obrazów
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Lokalizacja folderu z obrazami do augmentacji
image_directory = 'image_data/'
augmented_directory = 'augmented_data/'

# Lista obrazków w folderze
image_list = os.listdir(image_directory)

# Przeprowadzenie augmentacji dla każdego obrazka
for image_name in image_list:
    img_path = os.path.join(image_directory, image_name)
    img = load_img(img_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    num_augmentations = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_directory, save_prefix='aug', save_format='jpeg'):
        num_augmentations += 1
        if num_augmentations >= 5:  # Ograniczenie do 5 augmentacji na obraz
            break
