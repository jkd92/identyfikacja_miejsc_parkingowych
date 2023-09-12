import cv2
import numpy as np

# Lista do przechowywania wybranych punktów
points = []

def calculate_distance(point1, point2):
    """Oblicza odległość euklidesową pomiędzy dwoma punktami."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def click_event(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(image, f'({x}, {y})', (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if len(points) == 2:
            distance = calculate_distance(points[0], points[1])
            cv2.line(image, points[0], points[1], (255, 255, 255), 2)
            cv2.putText(image, f'Odleglosc: {distance:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            points.clear()
        
        cv2.imshow('Obraz', image)

# Wczytanie obrazu
image_path = 'sciezka_do_obrazu.jpg'
image = cv2.imread(image_path)

# Wyświetlenie obrazu
cv2.imshow('Obraz', image)

# Dodanie funkcji obsługującej zdarzenie kliknięcia
cv2.setMouseCallback('Obraz', click_event)

# Oczekiwanie na zamknięcie okna
cv2.waitKey(0)
cv2.destroyAllWindows()
