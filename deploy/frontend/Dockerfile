FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8001

WORKDIR /app

COPY deploy/frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/frontend/frontend.py frontend.py

EXPOSE 8001

CMD ["streamlit", "run", "frontend.py", "--server.port", "8001", "--server.address", "0.0.0.0"]
