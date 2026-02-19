import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Функция для обработки изображения
def process_blood_image(uploaded_file, mode):
    # Декодируем изображение
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_result = img.copy()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if mode == "Тромбоциты":
        # --- ВАШ КОД ДЛЯ ТРОМБОЦИТОВ ---
        lower_plt = np.array([130, 120, 70]) 
        upper_plt = np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower_plt, upper_plt)
        
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 15 < area < 300:
                count += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.circle(img_result, (int(x + w/2), int(y + h/2)), 10, (0, 255, 0), 2)
                cv2.putText(img_result, "PLT", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return img_result, {"Тромбоциты": count}

    else:
        # --- ВАШ КОД ДЛЯ НЕЙТРОФИЛОВ ---
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        counts = {"Юные": 0, "Палочки": 0, "Сегменты": 0}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400: continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h if w > h else float(h)/w
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0

            if solidity > 0.85 and circularity > 0.5:
                label, color = "Yuny", (0, 255, 255) # Желтый
                counts["Юные"] += 1
            elif circularity < 0.45 or aspect_ratio > 2.0:
                label, color = "P/Ya", (255, 0, 0) # Синий
                counts["Палочки"] += 1
            else:
                label, color = "S/Ya", (0, 255, 0) # Зеленый
                counts["Сегменты"] += 1

            cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return img_result, counts

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.title("🔬 Лабораторный Анализатор Крови")

# Выбор режима в боковой панели
mode = st.sidebar.radio("Выберите объект анализа:", ("Нейтрофилы", "Тромбоциты"))
uploaded_file = st.sidebar.file_uploader("Загрузите микрофотографию мазка", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Запуск обработки
    result_img, stats = process_blood_image(uploaded_file, mode)
    
    # Отображение результатов
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Визуализация")
        # Конвертируем BGR (OpenCV) в RGB (Streamlit)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("Результаты")
        if mode == "Тромбоциты":
            st.metric("Тромбоциты (PLT)", stats["Тромбоциты"])
        else:
            st.write(f"🟡 Юные: {stats['Юные']}")
            st.write(f"🔵 Палочкоядерные: {stats['Палочки']}")
            st.write(f"🟢 Сегментоядерные: {stats['Сегменты']}")
            total = sum(stats.values())
            st.divider()
            st.write(f"**Всего нейтрофилов: {total}**")
