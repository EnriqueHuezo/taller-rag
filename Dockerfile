FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema mínimas requeridas para compilar paquetes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Exponer el puerto de la API FastAPI
EXPOSE 8000

# Comando de inicio predeterminado
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
