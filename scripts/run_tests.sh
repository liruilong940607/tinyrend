#!/bin/bash

# Default values (matching CMakeLists.txt)
BUILD_CPP_ONLY="OFF"
SPECIFIC_TESTS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpp-only)
            BUILD_CPP_ONLY="ON"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [TEST_NAMES...]"
            echo "Options:"
            echo "  --cpp-only      Skip CUDA files and only build C++ files"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "TEST_NAMES:"
            echo "  Optional list of specific test executables to run"
            echo "  If not specified, all tests in ./build/tests will be run"
            echo ""
            echo "Examples:"
            echo "  $0                             # Run all tests"
            echo "  $0 --cpp-only                  # Build with C++ only and run all C++ tests"
            echo "  $0 camera/model util/se3       # Run only specified tests"
            echo "  $0 --cpp-only camera/model     # Build with C++ only and run specific C++ tests"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # Collect remaining arguments as test names
            SPECIFIC_TESTS+=("$1")
            shift
            ;;
    esac
done

FILE_DIR=$(dirname "$0")

# First build the project
if [ "$BUILD_CPP_ONLY" == "ON" ]; then
    bash $FILE_DIR/build.sh --cpp-only
else
    bash $FILE_DIR/build.sh
fi

# Run test executables and collect their exit codes
exit_code=0

if [ ${#SPECIFIC_TESTS[@]} -eq 0 ]; then
    # find all test executables in ./build/tests
    TEST_EXECUTABLES=$(find ./build/tests -type f -executable)
else
    # Only run specific tests
    TEST_EXECUTABLES=()
    for test in "${SPECIFIC_TESTS[@]}"; do
        TEST_EXECUTABLES+=("./build/tests/$test")
    done
fi

for test in "${TEST_EXECUTABLES[@]}"; do
    # echo "Running $test"
    eval "$test"
    test_exit=$?
    if [ $test_exit -ne 0 ]; then
        exit_code=$test_exit
    fi
done
printf "Exit code: %d\n" $exit_code

exit $exit_code 