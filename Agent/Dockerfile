FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5003

CMD ["gunicorn", "-w", "4", "--worker-class", "gthread", "--threads", "8", "--timeout", "120", "-b", "0.0.0.0:5003", "app:app"]
