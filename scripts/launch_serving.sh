#!/bin/bash
# Launch continuous-batching serving across all GPUs with nginx load balancer.
#
# Usage:
#   bash scripts/launch_serving.sh                    # 8 GPUs, default settings
#   bash scripts/launch_serving.sh --num-gpus 4       # 4 GPUs
#   bash scripts/launch_serving.sh --max-batch 32     # 32 concurrent per GPU
#   bash scripts/launch_serving.sh --source rl        # serve RL model
#
# This script:
# 1. Launches one FastAPI process per GPU (ports 8001-800N)
# 2. Generates nginx config dynamically
# 3. Starts nginx as the public-facing load balancer on port 80

set -e

# Defaults
NUM_GPUS=${NUM_GPUS:-8}
MAX_BATCH=${MAX_BATCH:-64}
SOURCE=${SOURCE:-sft}
MODEL_TAG=${MODEL_TAG:-}
STEP=${STEP:-}
BASE_PORT=8001
NGINX_PORT=80
TEMPERATURE=0.8
TOP_K=50
MAX_TOKENS=512
API_KEY=${BACKEND_API_KEY:-}

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --max-batch) MAX_BATCH="$2"; shift 2;;
        --source) SOURCE="$2"; shift 2;;
        --model-tag) MODEL_TAG="$2"; shift 2;;
        --step) STEP="$2"; shift 2;;
        --nginx-port) NGINX_PORT="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --top-k) TOP_K="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

PIDS=()
NGINX_CONF="/tmp/nanochat_nginx.conf"

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sudo nginx -s stop 2>/dev/null || true
    echo "All processes stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Generate nginx config
echo "Generating nginx config for $NUM_GPUS GPUs..."
cat > "$NGINX_CONF" << 'HEADER'
worker_processes auto;
error_log /tmp/nanochat_nginx_error.log warn;
pid /tmp/nanochat_nginx.pid;

events {
    worker_connections 4096;
}

http {
    # Use real client IP from X-Forwarded-For when behind a proxy (e.g. Vercel edge),
    # falling back to direct remote_addr for direct connections
    map $http_x_forwarded_for $rate_limit_key {
        ""      $binary_remote_addr;
        default $http_x_forwarded_for;
    }

    # Rate limiting: 10 req/s per real client IP with burst of 20
    limit_req_zone $rate_limit_key zone=chat_limit:10m rate=10r/s;
    # Max 4 concurrent connections per IP
    limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

    log_format main '$remote_addr - [$time_local] "$request" $status '
                    '$body_bytes_sent "$http_referer" '
                    'upstream=$upstream_addr rt=$request_time';
    access_log /tmp/nanochat_nginx_access.log main;

    upstream nanochat {
        least_conn;
HEADER

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    port=$((BASE_PORT + gpu))
    echo "        server 127.0.0.1:${port};" >> "$NGINX_CONF"
done

cat >> "$NGINX_CONF" << FOOTER
    }

    server {
        listen ${NGINX_PORT};

        # Rate limiting
        limit_req zone=chat_limit burst=20 nodelay;
        limit_conn conn_limit 4;

        # SSE-compatible proxy settings
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        location / {
            proxy_pass http://nanochat;
            proxy_read_timeout 300s;
        }

        # Chat completions endpoint — require API key if configured
        location /chat/completions {
FOOTER

# Conditionally add API key check
if [[ -n "$API_KEY" ]]; then
    cat >> "$NGINX_CONF" << FOOTER
            if (\$http_authorization != "Bearer ${API_KEY}") {
                return 401 '{"error": "Unauthorized"}';
            }
FOOTER
fi

cat >> "$NGINX_CONF" << FOOTER
            proxy_pass http://nanochat;
            proxy_read_timeout 300s;
        }

        location /health {
            proxy_pass http://nanochat;
            proxy_read_timeout 5s;
        }

        location /stats {
            proxy_pass http://nanochat;
            proxy_read_timeout 5s;
        }

        # Return 503 with useful message when rate limited
        error_page 503 = @rate_limited;
        location @rate_limited {
            default_type application/json;
            return 503 '{"error": "Rate limited. Please try again shortly."}';
        }
    }
}
FOOTER

echo "Nginx config written to $NGINX_CONF"

# Launch one FastAPI process per GPU
echo "Launching $NUM_GPUS worker processes..."
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    port=$((BASE_PORT + gpu))
    extra_args=""
    [[ -n "$MODEL_TAG" ]] && extra_args="$extra_args --model-tag $MODEL_TAG"
    [[ -n "$STEP" ]] && extra_args="$extra_args --step $STEP"

    echo "  GPU $gpu -> port $port (max_batch=$MAX_BATCH)"
    CUDA_VISIBLE_DEVICES=$gpu python -m scripts.chat_web_batch \
        --gpu-id 0 \
        --port "$port" \
        --source "$SOURCE" \
        --max-batch "$MAX_BATCH" \
        --temperature "$TEMPERATURE" \
        --top-k "$TOP_K" \
        --max-tokens "$MAX_TOKENS" \
        $extra_args &
    PIDS+=($!)
done

# Wait for workers to start
echo "Waiting for workers to initialize..."
sleep 10

# Start nginx
echo "Starting nginx on port $NGINX_PORT..."
sudo nginx -c "$NGINX_CONF"

echo ""
echo "========================================="
echo " NanoChat serving is running!"
echo " Public endpoint: http://localhost:${NGINX_PORT}"
echo " Workers: $NUM_GPUS GPUs, ${MAX_BATCH} batch/GPU"
echo " Total capacity: $((NUM_GPUS * MAX_BATCH)) concurrent requests"
if [[ -n "$API_KEY" ]]; then
    echo " API key auth: ENABLED"
else
    echo " API key auth: DISABLED (set BACKEND_API_KEY to enable)"
fi
echo "========================================="
echo ""
echo "Press Ctrl+C to stop all processes."

# Wait for any child to exit
wait -n "${PIDS[@]}" 2>/dev/null || true
echo "A worker process exited. Shutting down..."
cleanup
