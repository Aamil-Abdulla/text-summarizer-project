FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .
# Copy artifacts folder
COPY artifacts ./artifacts

# Expose the port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
