# Dockerfile
FROM python:3.11-slim
WORKDIR /app
# Dépendances système minimales (TF CPU a souvent besoin de libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
libgomp1 curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./app.py
COPY iris_ann_model.h5 ./iris_ann_model.h5
EXPOSE 5000
CMD ["python", "app.py"]