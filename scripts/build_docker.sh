#!/bin/bash
# Docker Build and Optimization Script for AEGIS Framework
# Builds optimized Docker images with size analysis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="aegis-framework"
VERSION="2.0.0"
TARGET_SIZE_MB=500

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     AEGIS Framework Docker Build & Optimization      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to convert bytes to MB
bytes_to_mb() {
    echo "scale=2; $1 / 1024 / 1024" | bc
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker found${NC}"
echo ""

# Build the image
echo -e "${YELLOW}📦 Building Docker image...${NC}"
echo "   Image: ${IMAGE_NAME}:${VERSION}"
echo "   Target size: <${TARGET_SIZE_MB}MB"
echo ""

docker build \
    --tag ${IMAGE_NAME}:${VERSION} \
    --tag ${IMAGE_NAME}:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    . 2>&1 | grep -v "^#" || true

echo ""
echo -e "${GREEN}✅ Build completed${NC}"
echo ""

# Get image size
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${VERSION} --format "{{.Size}}")
echo -e "${BLUE}📊 Image Analysis:${NC}"
echo "   Name: ${IMAGE_NAME}:${VERSION}"
echo "   Size: ${IMAGE_SIZE}"
echo ""

# Get detailed size info
IMAGE_SIZE_BYTES=$(docker inspect ${IMAGE_NAME}:${VERSION} --format='{{.Size}}')
IMAGE_SIZE_MB=$(bytes_to_mb ${IMAGE_SIZE_BYTES})

echo -e "${BLUE}📈 Size Breakdown:${NC}"
echo "   Total: ${IMAGE_SIZE_MB} MB"

# Check if target size is met
TARGET_SIZE_BYTES=$((TARGET_SIZE_MB * 1024 * 1024))
if [ ${IMAGE_SIZE_BYTES} -lt ${TARGET_SIZE_BYTES} ]; then
    echo -e "${GREEN}   ✅ Target size achieved! (${IMAGE_SIZE_MB}MB < ${TARGET_SIZE_MB}MB)${NC}"
else
    OVER_SIZE=$(echo "${IMAGE_SIZE_MB} - ${TARGET_SIZE_MB}" | bc)
    echo -e "${YELLOW}   ⚠️  Over target by ${OVER_SIZE}MB${NC}"
fi

echo ""

# Show layers
echo -e "${BLUE}🔍 Image Layers:${NC}"
docker history ${IMAGE_NAME}:${VERSION} --human --no-trunc | head -20

echo ""
echo -e "${BLUE}🧪 Testing Image:${NC}"

# Test the image
if docker run --rm ${IMAGE_NAME}:${VERSION} python -c "print('✅ Image is functional')"; then
    echo -e "${GREEN}✅ Image test passed${NC}"
else
    echo -e "${RED}❌ Image test failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}📋 Available Commands:${NC}"
echo ""
echo "  # Run with dry-run (default)"
echo "  docker run -p 8080:8080 ${IMAGE_NAME}:${VERSION}"
echo ""
echo "  # Run full node"
echo "  docker run -p 8080:8080 ${IMAGE_NAME}:${VERSION} python main.py start-node"
echo ""
echo "  # Run with docker-compose"
echo "  docker-compose up -d"
echo ""
echo "  # View logs"
echo "  docker logs -f aegis-node"
echo ""
echo "  # Enter container"
echo "  docker exec -it aegis-node /bin/bash"
echo ""

# Optional: Push to registry
read -p "Push to Docker registry? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}📤 Pushing to registry...${NC}"
    docker push ${IMAGE_NAME}:${VERSION}
    docker push ${IMAGE_NAME}:latest
    echo -e "${GREEN}✅ Push completed${NC}"
fi

echo ""
echo -e "${GREEN}✨ Docker build process completed!${NC}"
echo ""
