FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    bash \  
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

# Instalar paquetes de Python desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/app.py"]
