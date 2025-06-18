FROM python:3.9-slim

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Pull LFS files
RUN git lfs pull

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "api.py"]
