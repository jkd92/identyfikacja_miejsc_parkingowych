import numpy as np  # Importowanie biblioteki NumPy
import cv2  # Importowanie biblioteki OpenCV
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib

# Wczytywanie i konwersja obrazu na odcienie szarości
img = cv2.cvtColor(cv2.imread("C:/users/4P/Documents/police.jpg"), cv2.COLOR_BGR2GRAY)

# Parametry dla HOG
cell_size = (8, 8)  # Rozmiar komórki (wysokość x szerokość) w pikselach
block_size = (2, 2)  # Rozmiar bloku (wysokość x szerokość) w komórkach
nbins = 9  # Liczba przedziałów histogramu

# Inicjalizacja deskryptora HOG
hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

# Obliczenie cech HOG
ncells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])

# Obliczanie cech HOG dla obrazu
hog_feats = hog.compute(img)\
    .reshape(ncells[1] - block_size[1] + 1,
             ncells[0] - block_size[0] + 1,
             block_size[1], block_size[0], nbins)\
    .transpose((1, 0, 2, 3, 4))  # Transpozycja dla lepszego indeksowania

# Inicjalizacja macierzy gradientów i licznika komórek
gradients = np.zeros((ncells[0], ncells[1], nbins))
cell_count = np.full((ncells[0], ncells[1], 1), 0, dtype=int)

# Agregacja gradientów
for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y:ncells[0] - block_size[0] + off_y + 1,
                  off_x:ncells[1] - block_size[1] + off_x + 1] += \
            hog_feats[:, :, off_y, off_x, :]
        cell_count[off_y:ncells[0] - block_size[0] + off_y + 1,
                   off_x:ncells[1] - block_size[1] + off_x + 1] += 1

# Normalizacja gradientów
gradients /= cell_count

# Wyświetlenie obrazu i gradientów
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

bin = 5  # Wybór jednego przedziału histogramu
plt.pcolor(gradients[:, :, bin])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable="box")
plt.colorbar()
plt.show()
