# Use a lightweight official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies if necessary
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install Python packages including redis
RUN pip install --no-cache-dir polars pyarrow yfinance duckdb fastapi uvicorn pydantic numpy redis

# Copy the engine package, data directory, and API script into the container
COPY engine/ /app/engine/
COPY data/ /app/data/
COPY api.py /app/api.py

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]