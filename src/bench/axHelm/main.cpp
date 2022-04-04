#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"

#include "nrssys.hpp"
#include "setupAide.hpp"
#include "platform.hpp"
#include "configReader.hpp"

namespace {

occa::kernel axKernel;
occa::kernel refAxKernel;

occa::memory o_D; 
occa::memory o_S;    
occa::memory o_ggeo;
occa::memory o_q;    
occa::memory o_Aq;
occa::memory o_Aq_ref;
occa::memory o_lambda;
occa::memory o_exyz;
occa::memory o_gllwz;
occa::memory o_elementList;

int Np; 
int Ng = -1;
int Nelements; 
size_t wordSize = 8;

template<typename FloatingPointType>
double computeErrImpl(occa::memory o_Ax, occa::memory o_Ax_ref)
{
  std::vector<FloatingPointType> Ax(Nelements * Np, 0.0);
  std::vector<FloatingPointType> AxRef(Nelements * Np, 0.0);
  o_Ax.copyTo(Ax.data(), Nelements * Np * sizeof(FloatingPointType));
  o_Ax_ref.copyTo(AxRef.data(), Nelements * Np * sizeof(FloatingPointType));

  FloatingPointType error = 0.0;

  for(auto i = 0; i < Ax.size(); ++i){
    error = std::max(error, std::abs(Ax[i]-AxRef[i]));
  }

  double lInfError = static_cast<double>(error);
  MPI_Allreduce(MPI_IN_PLACE, &lInfError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return lInfError;
}

double computeError(occa::memory o_Ax, occa::memory o_Ax_ref){
  double lInfError = -100.0;
  if(wordSize == 4){
    lInfError = computeErrImpl<float>(o_Ax, o_Ax_ref);
  }
  else if(wordSize == 8){
    lInfError = computeErrImpl<double>(o_Ax, o_Ax_ref);
  }
  return lInfError;
}

std::pair<double,double> run(int Ntests, int BKmode, int Ndim, int computeGeom, bool performCorrectnessCheck)
{
  double error = -100.0;
  const int offset = Nelements * Np;
  const int loffset = 0;

  if(performCorrectnessCheck){
    if(!computeGeom){
      axKernel(Nelements, offset, loffset, o_elementList, o_ggeo, o_D, o_S, o_lambda, o_q, o_Aq);
      refAxKernel(Nelements, offset, loffset, o_elementList, o_ggeo, o_D, o_S, o_lambda, o_q, o_Aq_ref);
      error = computeError(o_Aq, o_Aq_ref);
    }
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for(int test = 0; test < Ntests; ++test) {
    if(computeGeom){
      axKernel(Nelements, offset, loffset, o_elementList, o_exyz, o_gllwz, o_D, o_S, o_lambda, o_q, o_Aq);
    } else {
      axKernel(Nelements, offset, loffset, o_elementList, o_ggeo, o_D, o_S, o_lambda, o_q, o_Aq);
    }
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  return std::make_pair((MPI_Wtime() - start) / Ntests, error);
} 

void* (*randAlloc)(int);

void* rand32Alloc(int N)
{
  float* v = (float*) malloc(N * sizeof(float));

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

void* rand64Alloc(int N)
{
  double* v = (double*) malloc(N * sizeof(double));

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

} // namespace

int main(int argc, char** argv)
{
  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  configRead(MPI_COMM_WORLD);
  std::string installDir(getenv("NEKRS_HOME"));
  setupAide options; 

  int err = 0;
  int cmdCheck = 0;

  std::string threadModel;
  int N;
  int Ndim = 1;
  int okl = 1;
  int BKmode = 0;
  int Ntests = -1;
  int computeGeom = 0;

  while(1) {
    static struct option long_options[] =
    {
      {"p-order", required_argument, 0, 'p'},
      {"g-order", required_argument, 0, 'g'},
      {"computeGeom", no_argument, 0, 'c'},
      {"block-dim", required_argument, 0, 'd'},
      {"elements", required_argument, 0, 'e'},
      {"backend", required_argument, 0, 'b'},
      {"arch", required_argument, 0, 'a'},
      {"bk-mode", no_argument, 0, 'm'},
      {"fp32", no_argument, 0, 'f'},
      {"help", required_argument, 0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long (argc, argv, "", long_options, &option_index);

    if (c == -1)
      break;

    switch(c) {
    case 'p':
      N = atoi(optarg); 
      cmdCheck++; 
      break;
    case 'g':
      Ng = atoi(optarg); 
      break;
    case 'c':
      computeGeom = 1; 
      break;
    case 'd':
      Ndim = atoi(optarg);
      break;
    case 'e':
      Nelements = atoi(optarg);
      cmdCheck++;
      break;
    case 'b':
      options.setArgs("THREAD MODEL", std::string(optarg));
      cmdCheck++;
      break;
    case 'm':
      BKmode = 1;
      break;
    case 'f':
      wordSize = 4;;
      break;
    case 'i':
      Ntests = atoi(optarg);
      break;
    case 'h':
      err = 1;
      break;
    default:
      err = 1;
    }
  }

  if(err || cmdCheck != 3) {
    if(rank == 0)
      printf("Usage: ./nekrs-axhelm  --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>\n"
             "                    [--block-dim <n>]\n"
             "                    [--g-order <n>] [--computeGeom]\n"
             "                    [--bk-mode] [--fp32] [--iterations <n>]\n"); 
    exit(1); 
  }

  if(Ng < 0) Ng = N; 
  Nelements = std::max(1, Nelements/size);
  constexpr int p_Nggeo {7};
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;
  const int Nq_g = Ng + 1;
  const int Np_g = Nq_g * Nq_g * Nq_g; 

  platform = platform_t::getInstance(options, MPI_COMM_WORLD, MPI_COMM_WORLD); 
  const int Nthreads =  omp_get_max_threads();

  // build+load kernel
  occa::properties props = platform->kernelInfo + meshKernelProperties(N);
  if(wordSize == 4) props["defines/dfloat"] = "float";
  if(Ng != N) {
    props["defines/p_Nq_g"] = Nq_g;
    props["defines/p_Np_g"] = Np_g;
  }
  if(BKmode) props["defines/p_poisson"] = 1;

  std::string kernelName = "elliptic";
  if(Ndim > 1) kernelName += "Block";
  kernelName += "PartialAx";
  if(!BKmode) kernelName += "Coeff";
  if(Ng != N) {
    if(computeGeom) {
      if(Ng == 1) {
        kernelName += "Trilinear";
      } else {
        printf("Unsupported g-order=%d\n", Ng);
        exit(1);
      }
    } else {
      printf("for now g-order != p-order requires --computeGeom!\n");
      exit(1);
      kernelName += "Ngeom";
    } 
  }
  kernelName += "Hex3D";
  if (Ndim > 1) kernelName += "_N" + std::to_string(Ndim);

  if(platform->device.mode() == "CUDA"){
    props["compiler_flags"] += " -O3 ";
    props["compiler_flags"] += " --ftz=true ";
    props["compiler_flags"] += " --prec-div=false ";
    props["compiler_flags"] += " --prec-sqrt=false ";
    props["compiler_flags"] += " --use_fast_math ";
    props["compiler_flags"] += " --fmad=true ";
    //props["compiler_flags"] += "-Xptxas -dlcm=ca";
  }

  if(1)
  if(platform->device.mode() == "HIP"){
    props["defines/OCCA_USE_HIP"] = 1;
    props["compiler_flags"] += " -O3 ";
    props["compiler_flags"] += " -ffp-contract=on ";  // was fast
    //    props["compiler_flags"] += " -funsafe-math-optimizations ";
    //    props["compiler_flags"] += " -ffast-math ";
    //    props["compiler_flags"] += " -fno-vectorize "; // THIS CAN SLOW THINGS DOWN
  }

  
  std::cout << props;

  
  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  const std::string fileName = 
    installDir + "/okl/elliptic/" + kernelName + ext;

  const std::string refKernelName = "referenceImplementation";
  {
    occa::properties kernelProps = props;
    kernelProps["defines/p_knl"] = -1;
    refAxKernel = platform->device.buildKernel(fileName, refKernelName, kernelProps);
  }
  
  // populate arrays
  randAlloc = &rand64Alloc; 
  if(wordSize == 4) randAlloc = &rand32Alloc;

  dlong* elementList = (dlong*) calloc(Nelements, sizeof(dlong));
  for(int e = 0; e < Nelements; ++e)
    elementList[e] = e;
  o_elementList = platform->device.malloc(Nelements * sizeof(dlong), elementList);
  free(elementList);

  void *DrV   = randAlloc(Nq * Nq);
  void *ggeo  = randAlloc(Np_g * Nelements * p_Nggeo);
  void *q     = randAlloc((Ndim * Np) * Nelements);
  void *Aq    = randAlloc((Ndim * Np) * Nelements);
  void *exyz  = randAlloc((3 * Np_g) * Nelements);
  void *gllwz = randAlloc(2 * Nq_g);

  o_D = platform->device.malloc(Nq * Nq * wordSize, DrV);
  free(DrV);
  o_S = o_D;
  o_ggeo = platform->device.malloc(Np_g * Nelements * p_Nggeo * wordSize, ggeo);
  free(ggeo);
  o_q = platform->device.malloc((Ndim * Np) * Nelements * wordSize, q);
  free(q);
  o_Aq = platform->device.malloc((Ndim * Np) * Nelements * wordSize, Aq);
  o_Aq_ref = platform->device.malloc((Ndim * Np) * Nelements * wordSize, Aq);
  free(Aq);
  o_exyz = platform->device.malloc((3 * Np_g) * Nelements * wordSize, exyz);
  free(exyz);
  o_gllwz = platform->device.malloc(2 * Nq_g * wordSize, gllwz);
  free(gllwz);

  void *lambda = randAlloc(2 * Np * Nelements);
  o_lambda = platform->device.malloc(2 * Np * Nelements * wordSize, lambda);
  free(lambda);

  int kernels[4] = {1,3,4,5};
  constexpr int Nkernels = 4;
  for(int ind=0;ind<Nkernels;++ind){
    int kernelNumber  = kernels[ind];

    // 3D kernel layout requires a sufficiently small order
    // so as to not exceed the maximum threads / thread block
    if(Np > 1024 && kernelNumber == 3) continue;
    if(Np > 1024 && kernelNumber == 7) continue;

    auto kernelProps = props;
    kernelProps["defines/p_knl"] = kernelNumber;
    axKernel = platform->device.buildKernel(fileName, kernelProps, true);
    
    // warm-up
    auto elapsedAndError = run(10, BKmode, Ndim, computeGeom, true);
    double elapsed = elapsedAndError.first;
    const double error = elapsedAndError.second;
    const int elapsedTarget = 10;
    if(Ntests < 0) Ntests = elapsedTarget/elapsed;

    // ***** 
    elapsedAndError = run(Ntests, BKmode, Ndim, computeGeom, false);
    // ***** 

    elapsed = elapsedAndError.first;
 
    // print statistics
    const dfloat GDOFPerSecond = (size * Nelements * Ndim * (N * N * N) / elapsed) / 1.e9;

    size_t bytesMoved = Ndim * 2 * Np * wordSize; // x, Ax
    bytesMoved += 6 * Np_g * wordSize; // geo
    if(!BKmode) bytesMoved += 3 * Np * wordSize; // lambda1, lambda2, Jw
    const double bw = (size * Nelements * bytesMoved / elapsed) / 1.e9;

    double flopCount = Np * 12 * Nq + 15 * Np;
    if(!BKmode) flopCount += 5 * Np;
    const double gflops = Ndim * (size * flopCount * Nelements / elapsed) / 1.e9;

    if(rank == 0)
      std::cout << "MPItasks=" << size
                << " OMPthreads=" << Nthreads
                << " NRepetitions=" << Ntests
                << " Ndim=" << Ndim
                << " N=" << N
                << " Ng=" << Ng
                << " Nelements=" << size * Nelements
                << " elapsed time=" << elapsed
                << " bkMode=" << BKmode
                << " wordSize=" << 8*wordSize
                << " error=" << error
                << " kernel=" << kernelNumber
                << " GDOF/s=" << GDOFPerSecond
                << " GB/s=" << bw
                << " GFLOPS/s=" << gflops
                << "\n";
  }

  MPI_Finalize();
  exit(0);
}
