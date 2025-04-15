# MPI-CUDA-LargeScaleKNN

This repo contains reference implementation(s) for the MPI- and
CUDA-accelerated computation of the k-nearest neighbor distnace for a
(large!) set of points: input points get distributed over possibly
many MPI ranks, each rank uses CUDA to do do k-d tree construction and
queries on that rank, and ranks exchange data as required to make sure
thatq every rank gets the right results even across rank boundaries.

In particular, this repo contains two different solutions for
different types of input configurations:

- `unorderedDataVariant` takes as input a single large file of float3
  points, and makes no assumptions whatsoever about the ordering of
  points within this file - points can be in any arbitrary random order.
  
- `prePartitionedDataVariant` assumes that whichever code generated
  the data points was using some sort of spatial partitioning; this
  code assumes that the input points are given in N separate files,
  and that the user will launch one rank per such input
  file. Different ranks' input files can still spatially overlap, but
  if if that overlap is less than completely random this code should
  still be more efficient than the completely unorderd variant.
  
Example calls:

- for unordered vairant, assuming `points.float3` is a binary file
  with N float3 formatted points:

```
mpirun -n <desiredNumRanks> ./cudaMpiKNN_unorderedData points.float3 -o distances.float -k 100 
```
(will write one output file with N float distances)

- for prepartitioned variant (assuming `fileNames.txt` is a text file
  containing `numFiles` file names:

  ```
mpirun -n <numFiles> ./cudaMpiKNN_prePartitionedData fileNames.txt -k 100 -o distancesOfRank
  ```
(will write one output file per rank, with prefix `distanceOfRank`)
  

# Dependencies:

- (as submodule) : cudaKDTree
- (as submodule) : cuBQL
- CUDA
- MPI
  
