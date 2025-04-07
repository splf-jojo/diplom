import torch
import cv2
import numpy as np
import gc
import os
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from constants import classes  # Предполагается, что у вас есть список классов

app = FastAPI()

MODEL_PATH = 'mvit32-2.pt'

# Параметры предобработки
MEAN = np.array([123.675, 116.28, 103.53])
STD = np.array([58.395, 57.12, 57.375])

# Количество кадров для модели и перекрытие
NUM_FRAMES = 32
OVERLAP = 16

# Загрузка модели при запуске приложения
print("Загрузка модели...")
model = torch.jit.load(MODEL_PATH)
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    device = torch.device('cuda')
    print("Используется GPU для вычислений.")
else:
    device = torch.device('cpu')
    print("Используется CPU для вычислений.")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def resize_and_pad(image, size=(224, 224)):
    h, w = image.shape[:2]
    scale = min(size[0]/h, size[1]/w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    delta_w = size[1] - new_w
    delta_h = size[0] - new_h
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)

    color = [114, 114, 114]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_and_pad(frame)
    frame = (frame - MEAN) / STD
    frame = np.transpose(frame, (2, 0, 1))
    return frame

def process_video_segments(frames, model):
    predictions = []
    num_frames = len(frames)
    start_idx = 0

    while start_idx < num_frames:
        end_idx = start_idx + NUM_FRAMES
        if end_idx > num_frames:
            end_idx = num_frames

        segment_frames = frames[start_idx:end_idx]

        if len(segment_frames) < NUM_FRAMES:
            last_frame = segment_frames[-1]
            while len(segment_frames) < NUM_FRAMES:
                segment_frames.append(last_frame)

        # Создаем входной тензор
        input_tensor = np.stack(segment_frames, axis=0)  # [NUM_FRAMES, 3, 224, 224]
        input_tensor = np.transpose(input_tensor, (1, 0, 2, 3))  # [3, NUM_FRAMES, 224, 224]
        input_tensor = input_tensor[None, None, ...]  # [1, 1, 3, NUM_FRAMES, 224, 224]
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32))

        input_tensor = input_tensor.to(device)

        # Предсказание
        with torch.no_grad():
            outputs = model(input_tensor)
            outputs = torch.nn.functional.softmax(outputs, dim=1)  # Получаем вероятности
            outputs = outputs.cpu().numpy()
            predicted_class_index = np.argmax(outputs, axis=1)[0]
            predicted_class = classes[predicted_class_index]
            predictions.append(predicted_class)
            print(f"Сегмент с кадра {start_idx} по {end_idx}: {predicted_class}")

        # Управление памятью
        del input_tensor
        del segment_frames
        gc.collect()

        # Обновляем индекс с учетом перекрытия
        if end_idx == num_frames:
            break  
        start_idx += NUM_FRAMES - OVERLAP

    return predictions

def post_process_predictions(predictions):
    processed = []
    prev_pred = None
    for pred in predictions:
        if pred != prev_pred and pred != '---':
            processed.append(pred)
            prev_pred = pred
    return processed

def combine_predictions(predictions):
    return ' '.join(predictions)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный видеофайл временно
        temp_video_path = f"/tmp/{file.filename}"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # Шаг 1: Извлечение кадров
        print("Извлечение кадров из видео...")
        frames = extract_frames(temp_video_path)
        print(f"Извлечено {len(frames)} кадров.")

        # Шаг 2: Предобработка кадров
        print("Предобработка кадров...")
        preprocessed_frames = []
        for i, frame in enumerate(frames):
            preprocessed_frame = preprocess_frame(frame)
            preprocessed_frames.append(preprocessed_frame)

            # Управление памятью
            del frame
            gc.collect()

        # Шаг 3: Обработка видео по сегментам и получение предсказаний
        print("Обработка видео по сегментам и получение предсказаний...")
        predictions = process_video_segments(preprocessed_frames, model)

        # Шаг 4: Пост-обработка предсказаний
        processed_predictions = post_process_predictions(predictions)
        print("Общий результат предсказаний после пост-обработки:")
        print(processed_predictions)

        # Шаг 5: Объединение предсказаний в текст
        final_text = combine_predictions(processed_predictions)
        print("Финальный результат:")
        print(final_text)

        # Удаляем временный видеофайл
        os.remove(temp_video_path)

        return JSONResponse(content={"predictions": processed_predictions, "final_text": final_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Запускаем приложение на указанном хосте и порту
    uvicorn.run(app, host="0.0.0.0", port=8000)
