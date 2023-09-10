import cv2  # Importowanie biblioteki OpenCV
import numpy as np  # Importowanie biblioteki NumPy
import matplotlib.pyplot as plt  # Importowanie biblioteki Matplotlib

# Wczytanie obrazu z dysku
img = cv2.imread("C:/Users/HP/Documents/police.jpg")

# Konwersja obrazu na format RGB
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Konwersja obrazu do skali szarości
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tworzenie wykresu z obrazami
plt.figure(figsize=(12, 10))  # Ustawienie rozmiaru wykresu

# Wyświetlenie obrazu w skali szarości
plt.subplot(121)  # Ustalenie, gdzie wyświetlić obraz (1 wiersz, 2 kolumny, 1 miejsce)
plt.imshow(gray_img, cmap='gray')
plt.title('Obraz w skali szarości (ang. Grayscale)')

# Wyświetlenie obrazu RGB
plt.subplot(122)  # Ustalenie, gdzie wyświetlić obraz (1 wiersz, 2 kolumny, 2 miejsce)
plt.imshow(rgb_img)
plt.title('Obraz RGB')

# Pokazanie obrazów
plt.show()
