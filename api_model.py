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

# Количество кадров для модели
NUM_FRAMES = 32

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

def extract_frames(video_path: str):
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
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [114, 114, 114]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image

def preprocess_frame(frame):
    # Переводим BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Изменяем размер и делаем паддинг
    frame = resize_and_pad(frame)
    # Нормализация
    frame = (frame - MEAN) / STD
    # Меняем оси: HWC -> CHW
    frame = np.transpose(frame, (2, 0, 1))
    return frame

def sample_or_pad_frames(frames: list, num_frames: int = NUM_FRAMES):
    """Функция, которая из набора кадров делает ровно num_frames кадров:
       - Если кадров меньше num_frames, дублируем последний кадр.
       - Если кадров больше num_frames, равномерно выбираем num_frames кадров."""
    length = len(frames)
    if length < num_frames:
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)
        return frames
    elif length > num_frames:
        indices = np.linspace(0, length - 1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]
        return frames
    else:
        return frames

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный видеофайл
        temp_video_path = f"/tmp/{file.filename}"
        with open(temp_video_path, "wb") as buffer:
            buffer.write(await file.read())

        # 1. Извлекаем кадры из видео
        print("Извлечение кадров из видео...")
        frames = extract_frames(temp_video_path)
        print(f"Извлечено {len(frames)} кадров.")

        if len(frames) == 0:
            return JSONResponse(
                content={"error": "Видео не содержит кадров или не может быть прочитано."},
                status_code=400
            )

        # 2. Предобрабатываем кадры
        preprocessed_frames = []
        for frame in frames:
            preprocessed_frames.append(preprocess_frame(frame))
            gc.collect()

        # 3. Формируем ровно NUM_FRAMES кадров
        final_frames = sample_or_pad_frames(preprocessed_frames, NUM_FRAMES)

        # 4. Готовим входной тензор для модели: [1, 1, 3, 32, 224, 224]
        input_tensor = np.stack(final_frames, axis=0)           # shape: [32, 3, 224, 224]
        input_tensor = np.transpose(input_tensor, (1, 0, 2, 3))   # shape: [3, 32, 224, 224]
        input_tensor = input_tensor[None, None, ...]              # shape: [1, 1, 3, 32, 224, 224]
        input_tensor = torch.from_numpy(input_tensor.astype(np.float32)).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  
            probabilities = probabilities.cpu().numpy()[0]  # Получаем массив вероятностей для каждого класса

        # 5. Сортировка вероятностей и выбор топ-10
        top_k = 10
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = [
            {"gesture": classes[idx], "probability": float(probabilities[idx])}
            for idx in top_indices
        ]

        print("Топ-10 предсказаний:")
        for pred in top_predictions:
            print(f"{pred['gesture']}: {pred['probability']:.4f}")

        # Удаляем временный файл
        os.remove(temp_video_path)

        # 6. Возвращаем результат
        return JSONResponse(
            content={"top_predictions": top_predictions}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
