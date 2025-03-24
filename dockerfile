FROM python:3.9-slim

# Install system dependencies (expand here)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cmake \
    pkg-config \
    git \
    ffmpeg \
    libfftw3-dev \
    libfftw3-single3 \
    libsamplerate0-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libtag1-dev \
    libyaml-dev \
    zlib1g-dev \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ ./src/
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/analyze.py"]
