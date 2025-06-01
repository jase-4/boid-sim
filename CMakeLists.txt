set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

cmake_minimum_required(VERSION 3.18)
project(boid_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (e.g., for RTX 2070 use 75)
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_language(CUDA)

# Include OpenCV directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/assets/textures)
include_directories(${CMAKE_SOURCE_DIR}/shaders)

# Gather source files
file(GLOB_RECURSE cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp"  "${CMAKE_SOURCE_DIR}/cuda/*.cpp" )
file(GLOB_RECURSE gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu" "${CMAKE_SOURCE_DIR}/cuda/*.cu")
file(GLOB_RECURSE gpu_header_files "${CMAKE_SOURCE_DIR}/src/*.cuh" "${CMAKE_SOURCE_DIR}/cuda/*.cuh")
#file(GLOB_RECURSE header_files "${CMAKE_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE header_files "${CMAKE_SOURCE_DIR}/include/*.h" "${CMAKE_SOURCE_DIR}/include/*.hpp" "${CMAKE_SOURCE_DIR}/cuda/*.hpp")

file(GLOB_RECURSE asset_files "${CMAKE_SOURCE_DIR}/assets/textures/*")
file(GLOB_RECURSE shader_files "${CMAKE_SOURCE_DIR}/shaders/*")



# Create executable
add_executable(boid_sim
    ${header_files}
    ${cpu_source_files}
    ${gpu_source_files}
    ${gpu_header_files}
)

set_target_properties(boid_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  # Enable RDC here too
)


# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)


# Link the libraries
target_link_libraries(boid_sim
  PRIVATE
  
  OpenGL::GL  # This is the correct target for OpenGL (on Windows, it links opengl32.lib)
  glad::glad
  assimp::assimp
  glfw


)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(boid_sim PRIVATE CUDA::cudart)



message(STATUS "OpenGL include dirs: ${OPENGL_INCLUDE_DIRS}")
message(STATUS "OpenGL libraries: ${OPENGL_LIBRARIES}")


# Link OpenCV to your project

# Compile features
target_compile_features(boid_sim PUBLIC cuda_std_17)
