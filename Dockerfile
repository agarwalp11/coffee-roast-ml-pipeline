
# Use Python 3.11 as the base environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into the container
COPY . .

# Document the API port
EXPOSE 8080

# Start FastAPI when the container runs
CMD ["sh", "-c", "exec python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
