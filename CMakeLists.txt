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
add_executable(cudaMpiKNN_unorderedData unorderedInputData.cu)
target_link_libraries(cukd_unorderedData PUBLIC cudaKDTree MPI::MPI_CXX)

add_executable(cudaMpiKNN_prePartitionedData samples/mpiHugeQuery.cu)
target_link_libraries(cudaMpiKNN_prePartitionedData PUBLIC cudaKDTree MPI::MPI_CXX)




# ==================================================================
# create _a lot_ of test cases: this generates the whole matrix of
# traversal_method x num_dims x {fcp,knn}
# ==================================================================
if (BUILD_ALL_TESTS)
  # test 2, 3, 4, and 8-dimensoinal data; the latter should - if it
  # works for N=8, work for any other N>4
  #  set(DIMS_TO_BUILD 3)
  set(DIMS_TO_BUILD 2 3 4 8)
  foreach (D IN ITEMS ${DIMS_TO_BUILD})
    # test all four possible traversal methosds
    foreach(method stackBased stackFree cct)
      # test knn queries, on regular trees (no explicit dimension per node)
      add_executable(cukd_float${D}-knn-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test knn queries, with 'explicit-dim' trees
      add_executable(cukd_float${D}-knn-${method}-xd testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-${method}-xd cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-${method}-xd
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DEXPLICIT_DIM=1
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, on regular trees
      add_executable(cukd_float${D}-fcp-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, with 'explicit-dim' trees
      add_executable(cukd_float${D}-fcp-${method}-xd testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-${method}-xd cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-${method}-xd
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DEXPLICIT_DIM=1
	-DTRAVERSAL_METHOD=${method})

    endforeach()


    foreach(method stackBased cct)
      # test knn queries, on regular trees (no explicit dimension per node)
      add_executable(cukd_float${D}-knn-spatial-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-knn-spatial-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-knn-spatial-${method}
	PUBLIC
	-DD_FROM_CMAKE=${D}
	-DSPATIAL=1
	-DUSE_KNN=1
	-DTRAVERSAL_METHOD=${method})

      # test fcp queries, on regular trees
      add_executable(cukd_float${D}-fcp-spatial-${method} testing/floatN-knn-and-fcp.cu)
      target_link_libraries(cukd_float${D}-fcp-spatial-${method} cudaKDTree)
      target_compile_definitions(cukd_float${D}-fcp-spatial-${method}
	PUBLIC
	-DSPATIAL=1
	-DD_FROM_CMAKE=${D}
	-DTRAVERSAL_METHOD=${method})

    endforeach()
  endforeach()
endif()


if (NOT CUKD_IS_SUBPROJECT)
  # add some unit tests
  include(CTest) 
  add_subdirectory(testing)
endif()
