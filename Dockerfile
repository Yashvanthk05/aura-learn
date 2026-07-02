FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install pip dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose the default port (can be overridden with PORT env var)
EXPOSE 8000

# Use sh -c so ${PORT} expansion works; default to 8000 if PORT not set
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]