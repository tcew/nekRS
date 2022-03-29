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

occa::kernel fdmKernel;

occa::memory o_Sx;
occa::memory o_Sy;
occa::memory o_Sz;
occa::memory o_invL;
occa::memory o_u;
occa::memory o_Su;

int Np; 
int Nelements; 

double run(int Ntests)
{
  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for(int test = 0; test < Ntests; ++test) {
    fdmKernel(Nelements, o_Su, o_Sx, o_Sy, o_Sz, o_invL, o_u);
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  return (MPI_Wtime() - start) / Ntests;
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

  int N;
  int okl = 1;
  int Ntests = -1;
  size_t wordSize = 8;

  while(1) {
    static struct option long_options[] =
    {
      {"p-order", required_argument, 0, 'p'},
      {"elements", required_argument, 0, 'e'},
      {"backend", required_argument, 0, 'b'},
      {"arch", required_argument, 0, 'a'},
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
    case 'e':
      Nelements = atoi(optarg);
      cmdCheck++;
      break;
    case 'b':
      options.setArgs("THREAD MODEL", std::string(optarg));
      cmdCheck++;
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
      printf("Usage: ./nekrs-fdm  --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>\n"
             "                    [--fp32] [--iterations <n>]\n"); 
    exit(1); 
  }

  Nelements = std::max(1, Nelements/size);
  const int Nq = N+1 ;
  const int Np = Nq * Nq * Nq;
  const int Nq_e = N+1;
  const int Np_e = Nq_e*Nq_e*Nq_e;

  platform = platform_t::getInstance(options, MPI_COMM_WORLD, MPI_COMM_WORLD); 
  const int Nthreads =  omp_get_max_threads();

  // build+load kernel
  occa::properties props = platform->kernelInfo + meshKernelProperties(N);
  if(wordSize == 4) props["defines/pfloat"] = "float";
  else props["defines/pfloat"] = "dfloat";

  props["defines/p_Nq_e"] = Nq_e;
  props["defines/p_Np_e"] = Np_e;
  props["defines/p_overlap"] = 0;

  // always benchmark ASM
  props["defines/p_restrict"] = 0;

  if(platform->device.mode() == "CUDA"){
    props["compiler_flags"] += " -O3 ";
    props["compiler_flags"] += " --ftz=true ";
    props["compiler_flags"] += " --prec-div=false ";
    props["compiler_flags"] += " --prec-sqrt=false ";
    props["compiler_flags"] += " --use_fast_math ";
    props["compiler_flags"] += " --fmad=true ";
    //props["compiler_flags"] += "-Xptxas -dlcm=ca";
  }
  
  if(platform->device.mode() == "HIP"){
    props["defines/OCCA_USE_HIP"] = 1;
    props["compiler_flags"] += " -O3 ";
    props["compiler_flags"] += " -ffp-contract=fast ";
    props["compiler_flags"] += " -funsafe-math-optimizations ";
    props["compiler_flags"] += " -ffast-math ";
    //props["compiler_flags"] += " -fno-vectorize "; THIS CAN SLOW THINGS DOWN
  }

  
  std::cout << props;
  
  std::string kernelName = "fusedFDM";
  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  const std::string fileName = 
    installDir + "/okl/elliptic/" + kernelName + ext;

  // populate arrays
  randAlloc = &rand64Alloc; 
  if(wordSize == 4) randAlloc = &rand32Alloc;

  void *Sx   = randAlloc(Nelements * Nq_e * Nq_e);
  void *Sy   = randAlloc(Nelements * Nq_e * Nq_e);
  void *Sz   = randAlloc(Nelements * Nq_e * Nq_e);
  void *invL = randAlloc(Nelements * Np_e);
  void *Su   = randAlloc(Nelements * Np_e);
  void *u    = randAlloc(Nelements * Np_e);
  void *tmp  = randAlloc(Nelements * Np_e);
  void *u0   = randAlloc(Nelements * Np_e);

  double *zero = (double*) calloc(Nelements * Np_e, sizeof(double));

  o_Sx = platform->device.malloc(Nelements * Nq_e* Nq_e * wordSize, Sx);
  free(Sx);
  o_Sy = platform->device.malloc(Nelements * Nq_e * Nq_e * wordSize, Sy);
  free(Sy);
  o_Sz = platform->device.malloc(Nelements * Nq_e * Nq_e * wordSize, Sz);
  free(Sz);
  o_invL = platform->device.malloc(Nelements * Np_e * wordSize, invL);
  free(invL);
  o_Su = platform->device.malloc(Nelements * Np_e * wordSize, Su);
  free(Su);
  o_u = platform->device.malloc(Nelements * Np_e * wordSize, u);
  //  free(u);

  int minKernel = 0, maxKernel = 11;
  for(int knl=minKernel;knl<=maxKernel;++knl){
    if(knl!=8){
    occa::properties saveprops = props;    
    saveprops["defines/p_knl"] = knl;
    fdmKernel = platform->device.buildKernel(fileName, saveprops, true);

    // checksum
    o_u.copyFrom(u);
    run(1);
    o_Su.copyTo(tmp);
    if(knl==0)
      o_Su.copyTo(u0);
    o_Su.copyFrom(zero);
    
    double checksum = 0;
    for(int n=0;n<Np_e*Nelements;++n){
      if(wordSize==4)
	checksum += fabs(((float*)tmp)[n]);
      else
	checksum += fabs(((double*)tmp)[n]);
    }
    
    // warm-up
    double elapsed = run(10);
    const int elapsedTarget = 10;
    if(Ntests < 0) Ntests = elapsedTarget/elapsed;
  
    // *****
    // warm up (for pci-e a100 that is slow to start)
    elapsed = run(1000);
    
    // actual run
    elapsed = run(Ntests);
    // ***** 
    
    // print statistics
    const dfloat GDOFPerSecond = (size * Nelements * (N* N * N) / elapsed) / 1.e9;
    
    size_t bytesPerElem = (3 * Np_e + 3 * Nq_e * Nq_e) * wordSize;
    const double bw = (size * Nelements * bytesPerElem / elapsed) / 1.e9;
    
    double flopsPerElem = 12 * Nq_e * Np_e + Np_e;
    const double gflops = (size * flopsPerElem * Nelements / elapsed) / 1.e9;
    
    if(rank == 0)
      //      std::cout << "MPItasks=" << size
      //		<< " OMPthreads=" << Nthreads
      std::cout << std::setprecision(5)
		<< " NRepetitions=" << Ntests
		<< " N=" << N
		<< " Nelements=" << size * Nelements
		<< " elapsed time=" << elapsed
		<< " wordSize=" << 8*wordSize
		<< " GDOF/s=" << GDOFPerSecond
		<< " GB/s=" << bw
		<< " GFLOPS/s=" << gflops
		<< " kernel=" << knl
		<< " checksum=" << std::setprecision(16) << checksum
		<< "\n";
    }
  }
  
  MPI_Finalize();
  exit(0);
}
