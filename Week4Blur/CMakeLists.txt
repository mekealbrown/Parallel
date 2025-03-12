# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(BlurProject C)

# Set C standard (optional, adjust as needed)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Common compiler flags as a list
set(COMMON_FLAGS -mavx2 -mfma -O2 -march=native -funroll-loops -g)
set(COMMON_LIBS m OpenMP::OpenMP_C)

# Source file
set(SOURCE_FILE blur.c)

# Define the different targets with their specific flags
add_executable(blur_unop ${SOURCE_FILE})
target_compile_options(blur_unop PRIVATE ${COMMON_FLAGS} -DUNOP)
target_link_libraries(blur_unop PRIVATE ${COMMON_LIBS})

add_executable(blur_op ${SOURCE_FILE})
target_compile_options(blur_op PRIVATE ${COMMON_FLAGS} -DOP)
target_link_libraries(blur_op PRIVATE ${COMMON_LIBS})

add_executable(blur_simd ${SOURCE_FILE})
target_compile_options(blur_simd PRIVATE ${COMMON_FLAGS} -DOP_SIMD)
target_link_libraries(blur_simd PRIVATE ${COMMON_LIBS})

add_executable(blur_all ${SOURCE_FILE})
target_compile_options(blur_all PRIVATE ${COMMON_FLAGS} -DUNOP -DOP -DOP_SIMD)
target_link_libraries(blur_all PRIVATE ${COMMON_LIBS})

# Custom clean target (optional, CMake already handles most cleanup)
add_custom_target(clean_extra
    COMMAND ${CMAKE_COMMAND} -E rm -f blur_* ../images/output.* result.txt *.png
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaned extra files"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning extra files not handled by CMake"
)