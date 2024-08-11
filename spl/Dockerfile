# Use an official Python runtime as a parent image
FROM anibali/pytorch:2.0.1-cuda11.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5002 for Flask (if you have a web server running)
EXPOSE 5002

# Default command to run in the container (optional, can be overridden by RunPod)
CMD ["python3"]
