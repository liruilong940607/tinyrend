cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Option to skip CUDA files and only build C++ files
option(BUILD_CPP_ONLY "Skip CUDA files and only build C++ files" OFF)

# Option to control whether to build tests
option(BUILD_TESTS "Build test executables" ON)

# Option to build examples
option(BUILD_EXAMPLES "Build example executables" ON)

# Option to build documentation
option(BUILD_DOCS "Build documentation" OFF)

# Link torch as a dependency
option(LINK_TORCH "Link torch as a dependency" OFF)

# Only include CUDA if not in CPP-only mode
if(NOT BUILD_CPP_ONLY)
    project(tinyrend LANGUAGES CUDA CXX)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
else()
    project(tinyrend LANGUAGES CXX)
endif()

if (LINK_TORCH)
    # Need to setup -DCMAKE_PREFIX_PATH=<path-to-libtorch> with cmake to find torch.
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# Header-only core
add_library(tinyrend INTERFACE)
target_include_directories(tinyrend INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

function(build_cuda_executables_recursive BASE_DIR PREFIX)
    # 1) grab all .cu/.cpp under BASE_DIR (recursively)
    if(BUILD_CPP_ONLY)
        file(GLOB_RECURSE ALL_SRCS
            "${BASE_DIR}/*.cpp"
        )
    else()
        file(GLOB_RECURSE ALL_SRCS
            "${BASE_DIR}/*.cu"
            "${BASE_DIR}/*.cpp"
        )
    endif()

    foreach(SRC ${ALL_SRCS})
        # 2) get path relative to BASE_DIR, e.g. "foo/bar/mytest.cpp"
        file(RELATIVE_PATH REL_PATH "${BASE_DIR}" "${SRC}")

        # 3) split into directory part ("foo/bar") and file stem ("mytest")
        get_filename_component(REL_DIR  "${REL_PATH}" PATH)
        get_filename_component(NAME_WE  "${REL_PATH}" NAME_WE)

        # 4) sanitize REL_DIR → replace "/" with "_"
        if(REL_DIR)
            string(REPLACE "/" "_" DIR_NAME "${REL_DIR}")
            set(TARGET_NAME "${PREFIX}_${DIR_NAME}_${NAME_WE}")
            set(OUT_DIR    "${CMAKE_BINARY_DIR}/${PREFIX}/${REL_DIR}")
        else()
            set(TARGET_NAME "${PREFIX}_${NAME_WE}")
            set(OUT_DIR    "${CMAKE_BINARY_DIR}/${PREFIX}")
        endif()

        add_executable(${TARGET_NAME} "${SRC}")
        target_link_libraries(${TARGET_NAME}
            PRIVATE tinyrend
        )
        if (LINK_TORCH)
            target_link_libraries(${TARGET_NAME}
                PRIVATE ${TORCH_LIBRARIES}
            )
        endif()
        set_target_properties(${TARGET_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${OUT_DIR}"
            OUTPUT_NAME               "${NAME_WE}"
        )
        message(STATUS 
            "  ↳ ${TARGET_NAME} → ${OUT_DIR}/${NAME_WE}"
        )
    endforeach()
endfunction()

# Build tests if enabled
if(BUILD_TESTS)
    build_cuda_executables_recursive(
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/tinyrend"
        tests
    )
endif()

# Build examples if enabled
if(BUILD_EXAMPLES)
    build_cuda_executables_recursive(
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        examples
    )
endif()

# Build documentation if enabled
if(BUILD_DOCS)
    add_subdirectory(docs)
endif()