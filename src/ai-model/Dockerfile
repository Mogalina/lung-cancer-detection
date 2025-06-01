FROM --platform=linux/arm64 python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY environment.yaml /app/

RUN conda env create -f environment.yaml

COPY app /app/app
COPY data /app/data
COPY app/configs/ /app/configs/
COPY saved-models /app/saved-models
COPY app/logs /app/logs

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "lung_cancer_env", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
