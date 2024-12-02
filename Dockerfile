# Базовый образ Python
FROM python:3.10-slim

# Установим рабочую директорию
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем весь проект в контейнер
COPY . .

# Открываем порт (необязательно, если Telegram работает только через API)
EXPOSE 8000

# Команда для запуска приложения
CMD ["python", "tgbot.py"]