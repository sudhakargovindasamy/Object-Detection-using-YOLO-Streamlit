FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app_streamlit.py .

# Download YOLO model on build (optional, can also happen at runtime)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
