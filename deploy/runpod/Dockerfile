FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
WORKDIR /app

# Install Python dependencies
COPY deploy/runpod/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy nanochat package (inference code)
COPY nanochat/ /app/nanochat/

# Download model weights at build time (baked in for FlashBoot cold starts)
ARG HF_REPO
ARG HF_TOKEN
ARG HF_STEP
COPY deploy/runpod/download_model.py .
RUN HF_TOKEN="$HF_TOKEN" python download_model.py --repo "$HF_REPO" --dest /app/model ${HF_STEP:+--step $HF_STEP}

# Copy handler
COPY deploy/runpod/handler.py .

CMD ["python3", "-u", "handler.py"]
