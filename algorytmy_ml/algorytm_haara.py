import numpy as np  # Importowanie biblioteki NumPy
import cv2 as cv  # Importowanie biblioteki OpenCV
import datetime  # Importowanie modułu datetime dla operacji związanych z czasem
import time  # Importowanie modułu time dla pomiaru czasu

start = time.time()  # Rozpoczęcie pomiaru czasu
now = datetime.datetime.now()  # Pobranie obecnej daty i czasu

# Wyświetlenie obecnego czasu
print("Obecny czas : ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Wczytanie kaskady Haar dla samochodów
car_cascade = cv.CascadeClassifier('cars.xml')

# Wczytanie i przekształcenie obrazu na odcienie szarości
img = cv.imread('vectra.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Wykrywanie samochodów na obrazie
cars = car_cascade.detectMultiScale(gray, 1.3, 5)

# Rysowanie prostokątnych ramek wokół wykrytych samochodów
for (x, y, w, h) in cars:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

end = time.time()  # Zakończenie pomiaru czasu
execution_time = (end - start)  # Obliczenie czasu wykonania

# Wyświetlenie czasu wykonania
print(f"Czas rozpoznania obiektu {execution_time}")

# Wyświetlenie obrazu z wykrytymi samochodami
cv.imshow('Car recognition Haar method', img)
cv.waitKey(0)
cv.destroyAllWindows()
