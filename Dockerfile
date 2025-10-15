# Dockerfile para AEGIS Framework - IA Distribuida y Colaborativa
# Programador Principal: Jose Gómez alias KaseMaster
# Contacto: kasemaster@aegis-framework.com
# Versión: 2.0.0
# Licencia: MIT

FROM python:3.13-slim

WORKDIR /app

# Instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copiar el proyecto
COPY . /app

# Variables y puerto por defecto para el dashboard
ENV AEGIS_DASHBOARD_PORT=8080 \
    AEGIS_LOG_LEVEL=INFO
EXPOSE 8080

# Comando por defecto: diagnóstico sin ejecutar servicios pesados
CMD ["python", "main.py", "--dry-run"]