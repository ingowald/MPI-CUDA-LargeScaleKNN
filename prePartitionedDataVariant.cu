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
#include "cukd/box.h"
#include <mpi.h>
#include <stdexcept>
#include <random>

using namespace cukd;

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
             std::vector<T> &values,
             std::string inFileName,
             int rank, int size,
             size_t *pBegin = 0,
             size_t *pNumTotal = 0
             )
{
  char temp[1024];
  sprintf(temp, "_%05d.part", rank);
  std::string full_filepath = inFileName + std::string(temp);

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
                         float *pMaxRadius,
                         int round)
{
#ifdef __CUDA_ARCH__
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  float3 qp = queries[tid];
  cukd::FlexHeapCandidateList cl(candidateLists+k*tid,k,
                                 round == 0 ? maxRadius : -1.f);
  cukd::stackFree::knn(cl,qp,tree,N);

  float myRadius = sqrtf(cl.maxRadius2());
  if (myRadius > *pMaxRadius)
    // ::atomicMax((int*)pMaxRadius,(int &)myRadius);
    cukd::atomicMax(pMaxRadius,myRadius);
#endif
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

std::vector<std::string> readListOfFileNames(std::string nameOfFileWithInFileNames)
{
  std::vector<std::string> names;
  std::ifstream in(nameOfFileWithInFileNames);
  while (in) {
    std::string line;
    std::getline(in,line);
    if (!in.good()) break;

    names.push_back(line.substr(0,line.find('\n')));
  }
  return names;
}

typedef cukd::box_t<float3> box3f;

bool allNegativeOnes(const std::vector<int> &vec)
{
  for (auto v : vec) if (v != -1) return false;
  return true;
}

std::vector<int> computePermutation(int rank, int size)
{
  std::srand(rank+0x1234567);
  for (int i=0;i<10;i++)
    std::rand();
  std::vector<int> ret(size);
  for (int i=0;i<size;i++) ret[i] = i;
  for (int i=size-1;i>0;--i) {
    int other = std::rand() % i;
    std::swap(ret[other],ret[i]);
  }
  return ret;
}

float computeDistance(const box3f &a,
                      const box3f &b)
{
  float3 diff = max(make_float3(0.f,0.f,0.f),max(a.lower-b.upper,b.lower-a.upper));
  return sqrtf(diff.x*diff.x+diff.y*diff.y+diff.z*diff.z);
}

int computeMyPeer(const box3f &myBounds,
                  const std::vector<box3f> &allBounds,
                  float cutOffRadius,
                  std::vector<bool> &alreadySeen,
                  const std::vector<int> &permutation)
{
  int best = -1;
  float closest = INFINITY;
  for (auto peer : permutation) {
    if (alreadySeen[peer]) continue;
    float dist = computeDistance(myBounds,allBounds[peer]);
    if (dist >= cutOffRadius) continue;
    if (dist >= closest) continue;
    closest = dist;
    best = peer;
  }
  return best;
}

int main(int ac, char **av)
{
  MPI_Init(&ac,&av);
  float maxRadius = std::numeric_limits<float>::infinity();
  int   k = 0;
  int   gpuAffinityCount = 0;
  std::string nameOfFileWithInFileNames;
  std::string outFilePrefix;

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-o")
      outFilePrefix = av[++i];
    else if (arg[0] != '-')
      nameOfFileWithInFileNames = arg;
    else if (arg == "-r")
      maxRadius = std::atof(av[++i]);
    else if (arg == "-g")
      gpuAffinityCount = std::atoi(av[++i]);
    else if (arg == "-k")
      k = std::atoi(av[++i]);
    else
      usage("unknown cmdline arg '"+arg+"'");
  }
  
  if (nameOfFileWithInFileNames.empty())
    usage("no input file name specified (should be a text file with list of input files)");
  if (outFilePrefix.empty())
    usage("no output file(s) prefix specified");
  if (k < 1)
    usage("no k specified, or invalid k value");
  
  // std::vector<std::string> inFileNames
  //   = readListOfFileNames(nameOfFileWithInFileNames);

  // -----------------------------------------------------------------------------
  // init mpi, check valid mpi size, and affinitize gpu (if desired)
  // -----------------------------------------------------------------------------
  MPIComm mpi(MPI_COMM_WORLD);
  // if (mpi.size != inFileNames.size())
  //   throw std::runtime_error("number of input files does not match MPI size");

  if (gpuAffinityCount) {
    int deviceID = mpi.rank % gpuAffinityCount;
    std::cout << "#" << mpi.rank << "/" << mpi.size
              << "setting active GPU #" << deviceID << std::endl;
    
    CUKD_CUDA_CALL(SetDevice(deviceID));
  }

  // -----------------------------------------------------------------------------
  // read our points (on host), and compute bounding box
  // -----------------------------------------------------------------------------
  double read_time = MPI_Wtime();

  size_t begin = 0;
  size_t numPointsTotal = 0;
  std::vector<float3> myPoints;
  //= readFilePortion<float3>(inFileNames[mpi.rank],0,1);
  readFileByName<float3>(myPoints,nameOfFileWithInFileNames,mpi.rank,mpi.size,&begin,&numPointsTotal);

  MPI_Barrier(mpi.comm);
  if (mpi.rank == 0)
      std::cout << "Elapsed read time: " << (MPI_Wtime() - read_time) << " seconds." << std::endl;

  double bounds_time = MPI_Wtime();

  box3f myBounds; myBounds.setEmpty();
  for (auto point : myPoints)
    myBounds.grow(point);
  {
    std::stringstream ss;
    ss << "#" << mpi.rank << "/" << mpi.size
       << ": got " << myPoints.size() << " points to work on, bounds is "
       << "("
       << myBounds.lower.x << ","
       << myBounds.lower.y << ","
       << myBounds.lower.z << ")-("
       << myBounds.upper.x << ","
       << myBounds.upper.y << ","
       << myBounds.upper.z << ")" 
       << std::endl;
    std::cout << ss.str();
  }

  MPI_Barrier(mpi.comm);
  if (mpi.rank == 0)
      std::cout << "Elapsed bounds time: " << (MPI_Wtime() - bounds_time) << " seconds." << std::endl;

  // -----------------------------------------------------------------------------
  // find out max num points anybody has, so we can allocate
  // ----------------------------------------------------------------------------- 
  size_t numPointsThatIHave = myPoints.size();
  size_t maxNumPointsAnybodyHas = 0;
  
  CUKD_MPI_CALL(Allreduce(&numPointsThatIHave,&maxNumPointsAnybodyHas,
                          1,MPI_LONG_LONG_INT,MPI_MAX,mpi.comm));
  
  // -----------------------------------------------------------------------------
  // allocate buffers (large enough to cycle), upload our data into
  // first of them, and build tree
  // -----------------------------------------------------------------------------
  double alloc1_time = MPI_Wtime();

  float3 *d_my_tree = 0;
  float3 *d_work_tree = 0;
  CUKD_CUDA_CALL(Malloc((void **)&d_my_tree,
                        maxNumPointsAnybodyHas*sizeof(myPoints[0])));
  CUKD_CUDA_CALL(Malloc((void **)&d_work_tree,
                        maxNumPointsAnybodyHas*sizeof(myPoints[0])));
  CUKD_CUDA_CALL(Memcpy(d_my_tree,myPoints.data(),
                        numPointsThatIHave*sizeof(myPoints[0]),
                        cudaMemcpyDefault));
  size_t N = numPointsThatIHave;

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed alloc (buildTree) time: " << (MPI_Wtime() - alloc1_time) << " seconds." << std::endl;

  // ---------------------------------------------------------------------------
  // build the tree:
  // ---------------------------------------------------------------------------
  double build_time = MPI_Wtime();

  cukd::buildTree(d_my_tree,N);

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed buildTree time: " << (MPI_Wtime() - build_time) << " seconds." << std::endl;

  // -----------------------------------------------------------------------------
  // upload our queries, and alloc candidate list(s)
  // -----------------------------------------------------------------------------
  double alloc2_time = MPI_Wtime();

  float3   *d_queries;
  size_t numQueries = myPoints.size();
  uint64_t *d_cand;
  CUKD_CUDA_CALL(Malloc((void **)&d_queries,numPointsThatIHave*sizeof(float3)));
  CUKD_CUDA_CALL(Memcpy(d_queries,myPoints.data(),numPointsThatIHave*sizeof(float3),
                        cudaMemcpyDefault));
  CUKD_CUDA_CALL(Malloc((void **)&d_cand,k*numPointsThatIHave*sizeof(uint64_t)));

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed alloc (queries) time: " << (MPI_Wtime() - alloc2_time) << " seconds." << std::endl;
  // -----------------------------------------------------------------------------
  // initialize algorithm for compuing per-rank send/recv peers
  // -----------------------------------------------------------------------------
  double perm_time = MPI_Wtime();

  std::vector<int> requestedByRank(mpi.size);
  std::vector<bool> alreadySeen(mpi.size); // == all false
  std::vector<box3f> allBounds(mpi.size);
  CUKD_MPI_CALL(Allgather(&myBounds,6,MPI_FLOAT,
                          allBounds.data(),6,MPI_FLOAT,mpi.comm));

  std::vector<int> allCounts(mpi.size);
  CUKD_MPI_CALL(Allgather(&numPointsThatIHave,1,MPI_INT,
                          allCounts.data(),1,MPI_INT,mpi.comm));

  float *d_maxRadius = 0;
  CUKD_CUDA_CALL(MallocManaged((void **)&d_maxRadius,sizeof(float)));
  std::vector<int> permutation = computePermutation(mpi.rank,mpi.size);

  MPI_Barrier(mpi.comm);

  if (mpi.rank == 0)
      std::cout << "Elapsed computePermutation time: " << (MPI_Wtime() - perm_time) << " seconds." << std::endl;
  // -----------------------------------------------------------------------------
  // now, do the queries and cycling:
  // -----------------------------------------------------------------------------
  double calc_time = MPI_Wtime();

  for (int round=0;round<mpi.size;round++) {
    CUKD_MPI_CALL(Barrier(mpi.comm));
    if (mpi.rank == 0) std::cout << "round " << round << std::endl;
    if (round == 0) {
      CUKD_CUDA_CALL(Memcpy(d_work_tree,d_my_tree,
                            numPointsThatIHave*sizeof(myPoints[0]),
                            cudaMemcpyDefault));
      alreadySeen[mpi.rank] = 1;
      N = numPointsThatIHave;
      // nothing to do, we already have our own tree
    } else {
      int myPeer = computeMyPeer(myBounds,allBounds,
                                 *d_maxRadius,alreadySeen,
                                 permutation);
      CUKD_MPI_CALL(Allgather(&myPeer,1,MPI_INT,
                              requestedByRank.data(),1,MPI_INT,mpi.comm));
      if (allNegativeOnes(requestedByRank))
        // we're ALL done!
        break;
      
      MPI_Request request;
      std::vector<MPI_Request> allRequests;
      if (myPeer == -1) {
        // nothing to receive
        N = 0;
      } else {
        N = allCounts[myPeer];
        CUKD_MPI_CALL(Irecv(d_work_tree,N*sizeof(float3),MPI_BYTE,myPeer,0,
                            mpi.comm,&request));
        allRequests.push_back(request);
        alreadySeen[myPeer] = 1;
      }

      for (int peer=0;peer<mpi.size;peer++) {
        int peerRequest = requestedByRank[peer];
        if (peerRequest != mpi.rank) continue;

        CUKD_MPI_CALL(Isend(d_my_tree,numPointsThatIHave*sizeof(float3),MPI_BYTE,
                            peer,0,mpi.comm,&request));
        allRequests.push_back(request);
      }
      CUKD_MPI_CALL(Waitall(allRequests.size(),allRequests.data(),MPI_STATUSES_IGNORE));
    }
    // -----------------------------------------------------------------------------
    if (N) {
      *d_maxRadius = 0.f;
      runQuery<<<divRoundUp(numQueries,1024ULL),1024ULL>>>
        (/* tree */d_work_tree,N,
         /* query params */d_cand,k,maxRadius,
         /* query points */d_queries,numQueries,
         d_maxRadius,round);
    }
    CUKD_CUDA_CALL(DeviceSynchronize());
  }
  
  //std::cout << "done all queries..." << std::endl;
  float *d_finalResults = 0;
  CUKD_CUDA_CALL(MallocManaged((void **)&d_finalResults,myPoints.size()*sizeof(float)));
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
      int gid = 960000/12*mpi.rank+i;
      if ((gid % 16*1024) == 0)
        printf("RES %012i = %f\n",gid,d_finalResults[i]);
    }
    MPI_Barrier(mpi.comm);
  }
#endif
  
  {
    char suffix[100];
    sprintf(suffix,"_%05d.float",mpi.rank);
    std::cout << "#" << mpi.rank << "/" << mpi.size << ": " << "Writing: " << outFilePrefix+suffix << std::endl;
    std::ofstream out(outFilePrefix+suffix,std::ios::binary);
    out.write((const char *)d_finalResults,numQueries*sizeof(float));
  }
  
  MPI_Barrier(mpi.comm);
  MPI_Finalize();
}
