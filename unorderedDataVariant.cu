// ======================================================================== //
// Copyright 2025-2025 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this fle except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cukd/cukd-math.h"
#include "cukd/traverse-stack-free.h"
#include "cukd/knn.h"
#include <mpi.h>
#include <stdexcept>

#define CUKD_MPI_CALL(fctCall)                                          \
  { int rc = MPI_##fctCall;                                             \
    if (rc != MPI_SUCCESS)                                              \
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+#fctCall); }

using cukd::divRoundUp;

struct MPIComm {
  MPIComm(MPI_Comm comm)
    : comm(comm)
  {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
  }
  MPI_Comm comm;
  int rank, size;
};

template<typename T>
std::vector<T> readFilePortion(std::string inFileName,
                               int rank, int size,
                               size_t *pBegin = 0,
                               size_t *pNumTotal = 0
                               )
{
  std::ifstream in(inFileName.c_str(),std::ios::binary);
  in.seekg(0,std::ios::end);
  size_t numBytes = in.tellg();
  in.seekg(0,std::ios::beg);

  size_t numData = numBytes / sizeof(T);
  if (pNumTotal) *pNumTotal = numData;
  size_t begin = numData * (rank+0)/size;
  if (pBegin) *pBegin = begin;
  size_t end   = numData * (rank+1)/size;
  in.seekg(begin*sizeof(T),std::ios::beg);
  
  std::vector<T> result(end-begin);
  in.read((char *)result.data(),(end-begin)*sizeof(T));
  return result;
}

template<typename T>
void readFileByName(
    std::vector<T>& values,
    std::string inFileName,
    int rank, int size,
    size_t* pBegin = 0,
    size_t* pNumTotal = 0
)
{
    char temp[1024];
    sprintf(temp, "_%05d.part", rank);
    std::string full_filepath = inFileName + std::string(temp);

    std::cout << "#" << rank << "/" << size << ": " << "Reading: " << full_filepath << std::endl;

    std::ifstream in(full_filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + full_filepath);
    }

    // Read the size of the data vector
    size_t dataSize;
    in.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

    std::cout << "#" << rank << "/" << size << ": " << "Data size: " << dataSize << std::endl;

    // Read the size of the name string
    size_t nameSize;
    in.read(reinterpret_cast<char*>(&nameSize), sizeof(nameSize));

    // Read the characters of the name string
    std::string name;
    name.resize(nameSize);
    in.read(&name[0], nameSize);
    std::cout << "#" << rank << "/" << size << ": " << "Name: " << name << std::endl;

    // Read 
    int num_comp = 0;
    in.read(reinterpret_cast<char*>(&num_comp), sizeof(num_comp));

    if (num_comp != 3) {
        throw std::runtime_error("Invalid number of components (3 for Position): " + std::to_string(num_comp));
    }

    // Read the size of the values vector
    size_t valuesSize;
    in.read(reinterpret_cast<char*>(&valuesSize), sizeof(valuesSize));
    std::cout << "#" << rank << "/" << size << ": " << "Position size: " << valuesSize / 3 << std::endl;

    // Read the values
    values.resize(valuesSize / 3);
    in.read(reinterpret_cast<char*>(values.data()), valuesSize * sizeof(float));
    in.close();
}

void usage(const std::string &error)
{
  std::cerr << "Error: " << error << std::endl << std::endl;
  std::cerr << "./mpiHugeQuery -k <k> [-r <maxRadius>] in.float3s -o out.dat" << std::endl;
  exit(error.empty()?0:1);
}



__global__ void runQuery(float3 *tree, int N,
                         uint64_t *candidateLists, int k, float maxRadius,
                         float3 *queries, int numQueries,
                         int round)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  float3 qp = queries[tid];
  cukd::FlexHeapCandidateList cl(candidateLists+k*tid,k,
                                 round == 0 ? maxRadius : -1.f);
  cukd::stackFree::knn(cl,qp,tree,N);
}

__global__ void extractFinalResult(float *d_finalResults,
                                   int numPoints,
                                   int k,
                                   uint64_t *candidateLists)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPoints) return;

  cukd::FlexHeapCandidateList cl(candidateLists+k*tid,k,-1.f);
  float result = cl.returnValue();
  if (!isinf(result))
    result = sqrtf(result);

  d_finalResults[tid] = result;
 }
  
int main(int ac, char **av)
{
  MPI_Init(&ac,&av);
  float maxRadius = std::numeric_limits<float>::infinity();
  int   k = 0;
  int   gpuAffinityCount = 0;
  std::string inFileName;
  std::string outFileName;

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o")
      outFileName = av[++i];
    else if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-r")
      maxRadius = std::atof(av[++i]);
    else if (arg == "-g")
      gpuAffinityCount = std::atoi(av[++i]);
    else if (arg == "-k")
      k = std::atoi(av[++i]);
    else
      usage("unknown cmdline arg '"+arg+"'");
  }

  if (inFileName.empty())
    usage("no input file name specified");
  if (outFileName.empty())
    usage("no output file name specified");
  if (k < 1)
    usage("no k specified, or invalid k value");

  MPIComm mpi(MPI_COMM_WORLD);
  if (gpuAffinityCount) {
    int deviceID = mpi.rank % gpuAffinityCount;
    std::cout << "#" << mpi.rank << "/" << mpi.size
              << "setting active GPU #" << deviceID << std::endl;
    CUKD_CUDA_CALL(SetDevice(deviceID));
  }else{
    CUKD_CUDA_CALL(SetDevice(0));
  }

  size_t begin = 0;
  size_t numPointsTotal = 0;

  // -----------------------------------------------------------------------------
  // read the data:
  // -----------------------------------------------------------------------------

  double read_time = MPI_Wtime();

  std::vector<float3> myPoints;
  //   = readFilePortion<float3>(inFileName,mpi.rank,mpi.size,&begin,&numPointsTotal);
  readFileByName<float3>(myPoints,inFileName,mpi.rank,mpi.size,&begin,&numPointsTotal);

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
    std::cout << "Elapsed read time: " << (MPI_Wtime() - read_time) << " seconds." << std::endl;

  std::cout << "#" << mpi.rank << "/" << mpi.size
            << ": got " << myPoints.size() << " points to work on"
            << std::endl;  

  // -----------------------------------------------------------------------------
  // find out max num points anybody has, so we can allocate
  // ----------------------------------------------------------------------------- 
  size_t numPointsThatIHave = myPoints.size();
  size_t N = numPointsThatIHave;
  size_t maxNumPointsAnybodyHas = 0;

  CUKD_MPI_CALL(Allreduce(&N, &maxNumPointsAnybodyHas, 1, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD));            

  double alloc1_time = MPI_Wtime();

  float3 *d_tree = 0;
  float3 *d_tree_recv = 0;
  //int N = myPoints.size();
  // alloc N+1 so we can store one more if anytoher rank gets oen more point
  CUKD_CUDA_CALL(Malloc((void **)&d_tree,(maxNumPointsAnybodyHas+1)*sizeof(float3)));
  CUKD_CUDA_CALL(Malloc((void **)&d_tree_recv,(maxNumPointsAnybodyHas+1)*sizeof(float3)));
  CUKD_CUDA_CALL(Memcpy(d_tree,myPoints.data(),N*sizeof(float3),
                        cudaMemcpyDefault));

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed alloc (buildTree) time: " << (MPI_Wtime() - alloc1_time) << " seconds." << std::endl;

  // ---------------------------------------------------------------------------
  // build the tree:
  // ---------------------------------------------------------------------------
  double build_time = MPI_Wtime();

  cukd::buildTree(d_tree,N);

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed buildTree time: " << (MPI_Wtime() - build_time) << " seconds." << std::endl;

  // ---------------------------------------------------------------------------
  // alloc2:
  // ---------------------------------------------------------------------------
  double alloc2_time = MPI_Wtime();

  float3   *d_queries;
  size_t numQueries = N;// myPoints.size();
  uint64_t *d_cand;
  CUKD_CUDA_CALL(Malloc((void **)&d_queries,N*sizeof(float3)));
  CUKD_CUDA_CALL(Memcpy(d_queries,myPoints.data(),N*sizeof(float3),cudaMemcpyDefault));
  CUKD_CUDA_CALL(Malloc((void **)&d_cand,N*k*sizeof(uint64_t)));

  float* d_finalResults = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_finalResults, numPointsThatIHave * sizeof(float)));

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed alloc (queries) time: " << (MPI_Wtime() - alloc2_time) << " seconds." << std::endl;
  
  // -----------------------------------------------------------------------------
  // now, do the queries and cycling:
  // -----------------------------------------------------------------------------
  double calc_time = MPI_Wtime();

  for (int round=0;round<mpi.size;round++) {
    
    if (round == 0) {
      // nothing to do , we already have our own tree
    } else {
      MPI_Request requests[2];
      int sendCount = N;
      int recvCount = 0;
      int sendPeer = (mpi.rank+1)%mpi.size;
      int recvPeer = (mpi.rank+mpi.size-1)%mpi.size;
      CUKD_MPI_CALL(Irecv(&recvCount,1*sizeof(int),MPI_BYTE,recvPeer,0,
                        mpi.comm,&requests[0]));
      CUKD_MPI_CALL(Isend(&sendCount,1*sizeof(int),MPI_BYTE,sendPeer,0,
                        mpi.comm,&requests[1]));
      CUKD_MPI_CALL(Waitall(2,requests,MPI_STATUSES_IGNORE));
      
      CUKD_MPI_CALL(Irecv(d_tree_recv,recvCount*sizeof(*d_tree),MPI_BYTE,recvPeer,0,
                          mpi.comm,&requests[0]));
      CUKD_MPI_CALL(Isend(d_tree,sendCount*sizeof(*d_tree),MPI_BYTE,sendPeer,0,
                          mpi.comm,&requests[1]));
      CUKD_MPI_CALL(Waitall(2,requests,MPI_STATUSES_IGNORE));
      
      N = recvCount;
      std::swap(d_tree,d_tree_recv);
    }
    // -----------------------------------------------------------------------------
    runQuery<<<divRoundUp(numQueries,1024ULL),1024ULL>>>
      (/* tree */d_tree,N,
       /* query params */d_cand,k,maxRadius,
       /* query points */d_queries,numQueries,
       round);
    CUKD_CUDA_CALL(DeviceSynchronize());
  }  

  //std::cout << "done all queries..." << std::endl;
  extractFinalResult<<<divRoundUp(numQueries,1024ULL),1024ULL>>>
    (d_finalResults,numQueries,k,d_cand);
  CUKD_CUDA_CALL(DeviceSynchronize());

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed queries time: " << (MPI_Wtime() - calc_time) << " seconds." << std::endl;

#if 0
  // output for probed verification
  for (int r=0;r<mpi.size;r++) {
    MPI_Barrier(mpi.comm);
    if (r == mpi.rank)
    for (int i=0;i<myPoints.size();i++) {
      int gid = begin+i;
      if ((gid % 16*1024) == 0)
        printf("RES %012i = %f\n",gid,d_finalResults[i]);
    }
    MPI_Barrier(mpi.comm);
  }
#endif

  // for (int i=0;i<mpi.size;i++) {
  //   MPI_Barrier(mpi.comm);
  //   if (i == mpi.rank) {
  //     FILE *file = fopen(outFileName.c_str(),i==0?"wb":"ab");
  //     fwrite(d_finalResults,sizeof(float),numQueries,file);
  //     fclose(file);
  //   }
  //   MPI_Barrier(mpi.comm);
  // }

  {
    char suffix[100];
    sprintf(suffix,"_%05d.float",mpi.rank);
    std::cout << "#" << mpi.rank << "/" << mpi.size << ": " << "Writing: " << outFileName+suffix << std::endl;
    std::ofstream out(outFileName+suffix,std::ios::binary);
    out.write((const char *)d_finalResults,numQueries*sizeof(float));
  }

  MPI_Barrier(mpi.comm);
  MPI_Finalize();
}
