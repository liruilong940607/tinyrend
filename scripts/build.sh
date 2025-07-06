#!/bin/bash

# Default values (matching CMakeLists.txt)
BUILD_CPP_ONLY="OFF"
BUILD_TESTS="ON"
BUILD_EXAMPLES="ON"
BUILD_DOCS="OFF"
LINK_TORCH="OFF"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            BUILD_CPP_ONLY="ON"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --docs)
            BUILD_DOCS="ON"
            shift
            ;;
        --link-torch)
            LINK_TORCH="ON"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cpp-only      Skip CUDA files and only build C++ files"
            echo "  --no-tests      Don't build test executables"
            echo "  --no-examples   Don't build example executables"
            echo "  --docs          Build documentation"
            echo "  --link-torch    Link torch as a dependency"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build the project
rm -rf build && mkdir -p build && cd build
cmake \
    -DBUILD_CPP_ONLY=$BUILD_CPP_ONLY \
    -DBUILD_TESTS=$BUILD_TESTS \
    -DBUILD_EXAMPLES=$BUILD_EXAMPLES \
    -DBUILD_DOCS=$BUILD_DOCS \
    -DLINK_TORCH=$LINK_TORCH \
    ..
if [ "$BUILD_DOCS" == "ON" ]; then
    make docs
else
    make -j12
fi

# Note: currently torch is not needed. Just for reference.
# wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.0%2Bcu121.zip
# unzip libtorch-cxx11-abi-shared-with-deps-2.5.0+cu121.zip
# cmake .. -DCMAKE_PREFIX_PATH="$(pwd)/../libtorch"
