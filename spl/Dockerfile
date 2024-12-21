# Base image with PyTorch and CUDA
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Update conda and install Python 3.11
RUN conda install -y python=3.11 && conda clean -afy

# Upgrade pip to ensure it works well with Python 3.11
RUN python -m pip install --upgrade pip

# (Optional) Remove any psutil artifacts pre-installed in the image
# - This step ensures that any old psutil .so files are removed.
RUN python -m pip uninstall -y psutil || true
RUN rm -f /opt/conda/lib/python3.11/site-packages/psutil/_psutil_linux*.so || true

# Reinstall psutil explicitly at a known version to avoid mismatched .so
RUN python -m pip install --no-cache-dir psutil==6.1.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/tmp

# Create and set the working directory
WORKDIR /app

# Ensure the data directory exists and is writable
RUN mkdir -p /app/data && chmod 777 /app/data

# Copy and install requirements using Python 3.11
COPY requirements.txt /app/requirements.txt

# Install all other Python packages. We ignore-installed to ensure pip
# re-resolves dependencies, but keep the pinned psutil from the prior step.
RUN python -m pip install --no-cache-dir --ignore-installed pyyaml \
    -r requirements.txt

# Expose the server port
EXPOSE 8001

# Define the command to run the server script
CMD ["python", "-m", "plugin_code.server"]
