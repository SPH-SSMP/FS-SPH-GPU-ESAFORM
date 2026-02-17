cmake_minimum_required(VERSION 3.11.3)

project(SPH LANGUAGES CUDA CXX)

# C++/CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA language
enable_language(CUDA)

# CUDA architectures and flags
set(CUDA_ARCHITECTURES "75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

# Enable separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)

# (Optional / legacy) Find CUDA toolkit if you rely on it elsewhere
find_package(CUDA QUIET REQUIRED)

# ----------------------------
# Fetch glm (header-only)
# ----------------------------
include(FetchContent)

# Pick a stable tag; adjust if you need a specific version
# Examples: 1.0.1, 0.9.9.8, etc.
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
)
FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
  # Add glm's CMake project to get its interface target
  add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# ----------------------------
# Sources
# ----------------------------
file(GLOB project_SRCS
    src/*.cpp
    src/Benchmarks/*.cpp
    src/Benchmarks/*.cu
    src/*.cu
)

# ----------------------------
# Target
# ----------------------------
add_executable(SPH ${project_SRCS})

# CUDA properties / arch
set_property(TARGET SPH PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(SPH PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

# Compile options
target_compile_options(SPH PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
    >
)

# CUDA include/lib paths (if you actually need them)
target_include_directories(SPH PRIVATE /usr/local/cuda-11.0/targets/x86_64-linux/include)
target_link_directories(SPH PRIVATE /usr/local/cuda-11.0/targets/x86_64-linux/lib)

# ----------------------------
# Link glm (header-only)
# ----------------------------
# glm defines an interface target; prefer glm::glm if available.
# The add_subdirectory above creates both 'glm' and the alias 'glm::glm'.
target_link_libraries(SPH PRIVATE glm::glm)
# (No extra include dirs needed; linking the interface target propagates includes.)

