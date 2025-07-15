#!/bin/bash

# Production deployment script

set -e  # Exit on any error

echo "Starting production deployment..."

# Update code
git pull origin main

# Build Docker image
docker build -t rag-chat:latest .

# Stop existing container
docker stop rag-chat || true
docker rm rag-chat || true

# Run new container
docker run -d \
  --name rag-chat \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/src/utils/vectorstore:/app/src/utils/vectorstore \
  -v $(pwd)/documents:/app/documents \
  rag-chat:latest

# Health check
sleep 10
if curl -f http://localhost:8000/health; then
    echo "Deployment successful!"
else
    echo "Deployment failed - health check failed"
    exit 1
fi

# Cleanup old images
docker image prune -f

echo "Production deployment completed!"
