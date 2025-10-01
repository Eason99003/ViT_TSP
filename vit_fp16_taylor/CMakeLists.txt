cmake_minimum_required(VERSION 3.10)
project(ViT C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")

# FP16 support option
option(USE_FP16 "Enable FP16 support for transformer encoder" OFF)
if(USE_FP16)
    add_compile_definitions(USE_FP16)
    message(STATUS "FP16 support enabled for transformer encoder")
endif()

# Include directories
include_directories(include)

# Find JPEG (optional)
find_package(JPEG QUIET)

# If system JPEG not found, try our custom installation
if(NOT JPEG_FOUND)
    set(JPEG_INCLUDE_DIRS "/tmp/jpeg/include")
    set(JPEG_LIBRARIES "/tmp/jpeg/lib/libjpeg.a")
    set(JPEG_FOUND TRUE)
endif()

# Source files for the original ViT implementation
set(SOURCES
    src/vit.c
    src/image_processing_improved.c
    src/model_loader_v2.c
    src/main.c
)

# Create executable
add_executable(vit ${SOURCES})

# Link libraries
if(JPEG_FOUND)
    target_link_libraries(vit ${JPEG_LIBRARIES})
    target_include_directories(vit PRIVATE ${JPEG_INCLUDE_DIRS})
    target_compile_definitions(vit PRIVATE HAVE_JPEG)
endif()

# Set output directory
set_target_properties(vit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create directory for output
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C flags: ${CMAKE_C_FLAGS}")

# Optional: Add install target
install(TARGETS vit DESTINATION bin)