# Use official Python base image
FROM python:3.10

WORKDIR /app

# Copy files
COPY app/ app/
COPY .env .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
