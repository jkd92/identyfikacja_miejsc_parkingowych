import cv2

# Lista do przechowywania wybranych punktów
points = []

def draw_rectangle(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Obraz', image)

        # Jeżeli wybrano dwa punkty, rysuj prostokąt
        if len(points) == 2:
            cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow('Obraz', image)
            points.clear()

# Wczytanie obrazu
image_path = 'sciezka_do_obrazu.jpg'
image = cv2.imread(image_path)

# Wyświetlenie obrazu
cv2.imshow('Obraz', image)

# Ustalenie funkcji callback dla zdarzeń myszy
cv2.setMouseCallback('Obraz', draw_rectangle)

# Oczekiwanie na zamknięcie okna
cv2.waitKey(0)
cv2.destroyAllWindows()
