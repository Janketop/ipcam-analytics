FROM python:3.11-slim

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --default-timeout=${PIP_DEFAULT_TIMEOUT} -r requirements.txt

COPY app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
