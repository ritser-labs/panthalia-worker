# ===== Stage 1: Builder Stage =====
FROM python:3.10-slim AS builder

# Set working directory and noninteractive mode for apt-get
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies into a virtual environment
COPY requirements.txt .
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# ===== Stage 2: Runtime Stage =====
FROM python:3.10-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv

COPY --from=builder /app/requirements.txt /app/requirements.txt

RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app /app

# Add these lines to create a writable state directory for the plugin
RUN mkdir -p /app/data/state && chown -R nobody:nogroup /app/data/state

ENV PATH="/venv/bin:$PATH"

EXPOSE 8001
CMD ["python", "-m", "plugin_code.server"]
