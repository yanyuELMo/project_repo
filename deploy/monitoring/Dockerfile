FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY deploy/monitoring/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/monitoring/app.py app.py
COPY deploy/monitoring/baseline.csv artifacts/baseline.csv

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
