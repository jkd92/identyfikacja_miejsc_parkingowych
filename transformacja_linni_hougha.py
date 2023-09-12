import cv2
import numpy as np

# Wczytanie obrazu
image = cv2.imread('car.jpg', cv2.IMREAD_COLOR)

# Konwersja do odcieni szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykrywanie krawędzi przy użyciu algorytmu Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Zastosowanie transformacji Hougha
# Parametry: obraz, odległość w pikselach, kąt w radianach, próg (liczba punktów na linii)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Rysowanie wykrytych linii na obrazie
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Wyświetlenie wynikowego obrazu
cv2.imshow('Wynik', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
