# CPU-only PyTorch. The default PyPI torch wheel drags in the CUDA runtime and
# adds well over a gigabyte to the image for hardware no container here has.
FROM python:3.13-slim

# librdkafka is a C library; confluent-kafka wheels bundle it, but gcc is still
# needed if pip has to fall back to a source build on this architecture.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached and only rebuilds when requirements
# change, so editing source code does not re-download PyTorch.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY stream/ ./stream/
COPY run.py .
COPY test_stream.py .

ENV PYTHONUNBUFFERED=1 \
    KAFKA_BOOTSTRAP=kafka:9092

# No CMD that runs the demo: the compose file decides what this image does, and
# the same image serves training, the demo, the API and the tests.
CMD ["python", "run.py", "--help"]
