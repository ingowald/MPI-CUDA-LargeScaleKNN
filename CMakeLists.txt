# ======================================================================== #
# Copyright 2025-2025 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0104 NEW)
set(CMAKE_BUILD_TYPE_INIT "Release")
project(MPI_CUDA_LargeScaleKNN VERSION 0.1.0 LANGUAGES C CXX)

# ==================================================================
# configure and enable cuda
# ==================================================================
if (CMAKE_CUDA_ARCHITECTURES)
  # set on the cmdline
  set (CUKD_INIT_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
elseif (${CMAKE_VERSION} VERSION_LESS 3.24)
  set (CUKD_INIT_ARCHS "70;80")
else()
  set (CUKD_INIT_ARCHS "all-major")
endif()
set(CUKD_CUDA_ARCHITECTURES "${CUKD_INIT_ARCHS}"
  CACHE STRING "CUDA Arch(s) to build against")
set(CMAKE_CUDA_ARCHITECTURES ${CUKD_CUDA_ARCHITECTURES})
enable_language(CUDA)

# ==================================================================
# require mpi
# ==================================================================
find_package(MPI REQUIRED)

# ==================================================================
# submodules
# ==================================================================
add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
add_subdirectory(submodules/cudaKDTree EXCLUDE_FROM_ALL)

# ==================================================================
# actual code
# ==================================================================
add_executable(cudaMpiKNN_unorderedData unorderedDataVariant.cu)
target_link_libraries(cudaMpiKNN_unorderedData PUBLIC cudaKDTree MPI::MPI_CXX)

add_executable(cudaMpiKNN_prePartitionedData prePartitionedDataVariant.cu)
target_link_libraries(cudaMpiKNN_prePartitionedData PUBLIC cudaKDTree MPI::MPI_CXX)


