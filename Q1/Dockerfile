# Use the official Hugging Face PyTorch GPU image as base
FROM huggingface/transformers-pytorch-gpu:latest

# Install additional dependencies: sacrebleu for BLEU score, gdown for Google Drive download,
# and striprtf to extract plain text from RTF files.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip install sacrebleu gdown striprtf

# Set the working directory inside the container
WORKDIR /app

# Copy the Python scripts into the container
COPY translate.py .
COPY evaluate.py .

# Default command: run translation then evaluation
CMD ["sh", "-c", "python translate.py && python evaluate.py"]