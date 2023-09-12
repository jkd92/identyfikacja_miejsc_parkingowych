import cv2

# Zmienne globalne
drawing = False  # flaga oznaczająca czy rysowanie jest aktywne
ix, iy = -1, -1  # inicjalne koordynaty punktu początkowego

# Funkcja obsługująca zdarzenia myszy
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', image)

# Wczytanie obrazu
image = cv2.imread('sciezka_do_obrazu.jpg')

# Tworzenie okna
cv2.namedWindow('image')

# Przypisanie funkcji obsługującej zdarzenia myszy
cv2.setMouseCallback('image', draw_rectangle)

while(1):
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Zapisanie obrazu
cv2.imwrite('image_with_rectangle.jpg', image)
