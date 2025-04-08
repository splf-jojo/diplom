# Сервис распознавания жестов

Сервис принимает видео с жестами и возвращает распознанные слова.

## 📌 Основное

-   **URL**: `http://<адрес_сервера>:8000/predict`
-   **Метод**: `POST`
-   **Контент**: `multipart/form-data`
-   **Поле**: `file` (видеофайл)

## ✅ Ответ

-   **Код 200** – успешная обработка:
    ```json
    {
        "predictions": ["привет", "как", "дела"],
        "final_text": "привет как дела"
    }
    ```
-   **Код 500** – ошибка:
    ```json
    {
        "error": "Описание ошибки"
    }
    ```

## 🚀 Запуск

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🧪 Тестирование API (пример на Python)

```python
import requests

files = {'file': open('demo_video.mp4', 'rb')}
res = requests.post('http://localhost:8000/predict', files=files)
print(res.json())
```

---
