# Base image with PyTorch and CUDA
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Update conda and install Python 3.11
RUN conda install -y python=3.11 && conda clean -afy

# Upgrade pip to ensure it works well with Python 3.11
RUN python -m pip install --upgrade pip

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/tmp

# Create and set the working directory
WORKDIR /app

# Copy and install requirements using Python 3.11
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --ignore-installed pyyaml \
    -r requirements.txt

# Expose the server port
EXPOSE 8001

# Define the command to run the server script
CMD ["python", "-m", "plugin_code.server"]
