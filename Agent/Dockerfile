FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5003

# The launcher keeps the development SQLite saver single-process and only
# enables the production four-worker default after PostgreSQL validation.
CMD ["python", "scripts/start_server.py"]
