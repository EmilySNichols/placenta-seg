
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/EmilySNichols/placenta-seg"
LABEL org.opencontainers.image.description="Placenta-Seg: nnU-Net v2 placenta segmentation"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nnUNet_results=/opt/nnUNet_results \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed

WORKDIR /app

# System libraries used by imaging dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install CPU-only PyTorch to avoid unnecessary CUDA libraries
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY run.py /app/run.py

# Download the versioned pretrained model
RUN curl -fL --retry 3 \
    "https://github.com/EmilySNichols/placenta-seg/releases/download/v1.0.1/placenta_nnunet.zip" \
    -o /tmp/placenta_nnunet.zip && \
    mkdir -p "$nnUNet_results" && \
    nnUNetv2_install_pretrained_model_from_zip \
        /tmp/placenta_nnunet.zip && \
    rm /tmp/placenta_nnunet.zip

ENTRYPOINT ["python", "/app/run.py"]

