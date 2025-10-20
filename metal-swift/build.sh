#!/bin/bash
#
# Build script for MetalSwiftBench
# Compiles the Swift implementation of GPU Audio Benchmarks
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MetalSwiftBench...${NC}"

# Create build directory
mkdir -p build

# Find all Swift files
SWIFT_FILES=$(find MetalSwiftBench -name "*.swift" | tr '\n' ' ')

# Find all Metal files
METAL_FILES=$(find MetalSwiftBench/Metal -name "*.metal" | tr '\n' ' ')

# Compile Metal shaders first
echo -e "${YELLOW}Compiling Metal shaders...${NC}"
for metal_file in $METAL_FILES; do
    echo "  Compiling $(basename $metal_file)"
    xcrun -sdk macosx metal -c "$metal_file" -o "build/$(basename $metal_file .metal).air"
done

# Create metallib
echo -e "${YELLOW}Creating Metal library...${NC}"
xcrun -sdk macosx metallib build/*.air -o build/default.metallib

# Compile Swift code
echo -e "${YELLOW}Compiling Swift code...${NC}"
swiftc $SWIFT_FILES \
    -import-objc-header MetalSwiftBench/MetalSwiftBench-Bridging-Header.h \
    -framework Metal \
    -framework CoreGraphics \
    -framework Foundation \
    -framework Accelerate \
    -o build/MetalSwiftBench \
    -O

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
    echo "Executable: build/MetalSwiftBench"
    
    
    echo -e "\nUsage: ./build/MetalSwiftBench --help"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi