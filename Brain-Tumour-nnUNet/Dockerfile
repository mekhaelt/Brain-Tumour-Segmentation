# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install nnUNetv2 and its dependencies
RUN pip3 install --no-cache-dir \
    nnunetv2 \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Create directory for model files
RUN mkdir -p /app/model

# Copy only the necessary model files
COPY Brain-Tumour-nnUNet/nnUNet_results/Dataset226_BraTS2024-BraTS-GLI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth /app/model/
COPY Brain-Tumour-nnUNet/nnUNet_results/Dataset226_BraTS2024-BraTS-GLI/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json /app/model/

# Set environment variables for nnUNetv2
ENV nnUNet_raw="/app/data/nnUNet_raw"
ENV nnUNet_preprocessed="/app/data/nnUNet_preprocessed"
ENV nnUNet_results="/app/model"

# Set the entrypoint to use nnUNetv2 inference
ENTRYPOINT ["nnUNetv2_predict"]

# To run the container, use:
# docker run -v "$(pwd)/test_images:/input" -v "$(pwd)/results:/output" brain-tumor-nnunet \
#     -i /input \
#     -o /output \
#     -d 226 \
#     -c nnUNetTrainer__nnUNetPlans__3d_fullres
