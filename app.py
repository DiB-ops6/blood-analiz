import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Функция обработки
def analyze_blood(image_file, mode):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if mode == "Нейтрофилы":
        lower, upper = np.array([120, 40, 40]), np.array([175, 255, 255])
    else: # Тромбоциты
        lower, upper = np.array([120, 20, 80]), np.array([175, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res_counts = {"Юные": 0, "Палочки": 0, "Сегменты": 0, "Тромбоциты": 0}
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20: continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        if mode == "Нейтрофилы" and area > 300:
            p = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (p*p) if p > 0 else 0
            sol = area / cv2.contourArea(cv2.convexHull(cnt)) if area > 0 else 0
            
            if sol > 0.85 and circ > 0.5:
                label, col = "Yuny", (0, 255, 255)
                res_counts["Юные"] += 1
            elif circ < 0.45 or (max(w,h)/min(w,h) > 2):
                label, col = "P/Ya", (255, 0, 0)
                res_counts["Палочки"] += 1
            else:
                label, col = "S/Ya", (0, 255, 0)
                res_counts["Сегменты"] += 1
            cv2.rectangle(img, (x,y), (x+w, y+h), col, 3)
        
        elif mode == "Тромбоциты" and area < 300:
            res_counts["Тромбоциты"] += 1
            cv2.circle(img, (int(x+w/2), int(y+h/2)), 10, (255, 0, 255), 2)
            
    return img, res_counts

# Интерфейс
st.title("🩸 Анализатор клеток крови")
mode = st.radio("Выберите объект анализа:", ["Нейтрофилы", "Тромбоциты"])
file = st.file_uploader("Загрузите фото мазка", type=['jpg', 'png', 'jpeg'])

if file:
    processed, counts = analyze_blood(file, mode)
    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption="Результат")
    st.write("### Результаты подсчета:")
    st.write(counts)
