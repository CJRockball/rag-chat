#!/bin/bash

set -e

echo "🚀 Starting production deployment..."

# Verify files exist on host before building
if [ ! -f "documents/Tender_Specs_ECDC2024OP0017_V1.pdf" ]; then
    echo "❌ Error: PDF file not found on host"
    exit 1
fi

# Build Docker image (documents will be copied in)
docker build -t rag-chat:latest .

# Stop existing container
docker stop rag-chat || true
docker rm rag-chat || true

# Run new container without document volume mount
docker run -d \
  --name rag-chat \
  --restart unless-stopped \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/src/utils/vectorstore:/app/src/utils/vectorstore" \
  rag-chat:latest

# Health check
sleep 15
if curl -f http://localhost:8000/health; then
    echo "✅ Deployment successful!"
else
    echo "❌ Deployment failed"
    docker logs rag-chat
    exit 1
fi
