# Dockerfile

# Use the base PyTorch image
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TMPDIR=/tmp

# Create and set the working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir --ignore-installed pyyaml \ 
    -r requirements.txt

# Expose the server port
EXPOSE 8000

# Define the command to run the server script
CMD ["python", "-m", "plugin_code.server"]
