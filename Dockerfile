FROM python:3.10-slim

# Avoid buffer and bytecode creation
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/huggingface


# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

# Copy project files
COPY . .

# COPY .env .  # or COPY .env /app/ if you set WORKDIR /app
# Collect static files
RUN python manage.py collectstatic --noinput


# Launch via Gunicorn
CMD ["gunicorn", "--timeout", "120","MyProject.wsgi:application", "--bind", "0.0.0.0:8000"]
