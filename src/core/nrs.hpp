#if !defined(nekrs_nekrs_hpp_)
#define nekrs_nekrs_hpp_

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <getopt.h>
#include <vector>

#include "nrssys.hpp"
#include "mesh3D.h"
#include "elliptic.h"
#include "cds.hpp"
#include "linAlg.hpp"
#include "timer.hpp"
#include "platform.hpp"

struct nrs_t
{
  int dim, elementType;

  mesh_t* _mesh;
  mesh_t* meshV;

  elliptic_t* uSolver;
  elliptic_t* vSolver;
  elliptic_t* wSolver;
  elliptic_t* uvwSolver;
  elliptic_t* pSolver;
  elliptic_t* meshSolver;

  cds_t* cds;

  oogs_t* gsh;

  dlong ellipticWrkOffset;

  int flow;

  int Nscalar;
  dlong fieldOffset;
  dlong cubatureOffset;
  setupAide vOptions, pOptions, mOptions;

  int NVfields, NTfields;

  int converged;

  dfloat dt[3], idt;
  dfloat p0th[3] = {0.0, 0.0, 0.0};
  dfloat CFL;
  dfloat unitTimeCFL;

  dfloat dp0thdt;
  int tstep;
  int lastStep;
  dfloat g0, ig0;

  int cht;

  int nEXT;
  int nBDF;
  int isOutputStep;
  int outputForceStep;

  dfloat* U, * P;
  dfloat* BF, * FU;

  // unit normal flow direction for constant flow rate
  dfloat flowDirection[3];
  int fromBID;
  int toBID;
  dfloat flowRate;

  //RK Subcycle Data
  int nRK;
  dfloat* coeffsfRK, * weightsRK, * nodesRK;
  occa::memory o_coeffsfRK, o_weightsRK;

  //ARK data
  int Nrk;
  dfloat* rkC;

  //EXTBDF data
  dfloat* coeffEXT, * coeffBDF, * coeffSubEXT;

  int Nsubsteps;
  dfloat* Ue, sdt;
  occa::memory o_Ue;

  dfloat* div;
  occa::memory o_div;

  dfloat rho, mue;
  occa::memory o_rho, o_mue;
  occa::memory o_meshRho, o_meshMue;

  dfloat* usrwrk;
  occa::memory o_usrwrk;

  occa::memory o_idH; // i.e. inverse of 1D Gll Spacing for quad and Hex

  int filterNc; // filter cut modes i.e. below is not touched
  dfloat* filterM, filterS;
  occa::memory o_filterMT; // transpose of filter matrix
  occa::kernel filterRTKernel; // Relaxation-Term based filtering
  occa::kernel advectMeshVelocityKernel;

  occa::kernel pressureAddQtlKernel;
  occa::kernel pressureStressKernel;

  occa::kernel subCycleVolumeKernel,  subCycleCubatureVolumeKernel;
  occa::kernel subCycleSurfaceKernel, subCycleCubatureSurfaceKernel;
  occa::kernel subCycleRKUpdateKernel;
  occa::kernel extrapolateKernel;
  occa::kernel subCycleRKKernel;
  occa::kernel subCycleInitU0Kernel;
  occa::kernel nStagesSum3Kernel;

  occa::kernel wgradientVolumeKernel;

  occa::kernel subCycleStrongCubatureVolumeKernel;
  occa::kernel subCycleStrongVolumeKernel;

  occa::kernel computeFaceCentroidKernel;
  occa::kernel computeFieldDotNormalKernel;

  occa::memory o_U, o_P;

  occa::memory o_Uc, o_Pc;
  occa::memory o_prevProp;

  occa::memory o_relUrst;
  occa::memory o_Urst;
  occa::kernel UrstCubatureKernel;
  occa::kernel UrstKernel;

  occa::memory o_BF;
  occa::memory o_FU;

  dfloat* prop;
  occa::memory o_prop, o_ellipticCoeff;

  //EXTBDF data
  occa::memory o_coeffEXT, o_coeffBDF, o_coeffSubEXT;

  occa::kernel advectionVolumeKernel;
  occa::kernel advectionCubatureVolumeKernel;

  occa::kernel strongAdvectionVolumeKernel;
  occa::kernel strongAdvectionCubatureVolumeKernel;

  occa::kernel gradientVolumeKernel;

  occa::kernel wDivergenceVolumeKernel;
  occa::kernel divergenceVolumeKernel;
  occa::kernel divergenceSurfaceKernel;

  occa::kernel divergenceStrongVolumeKernel;
  occa::kernel sumMakefKernel;
  occa::kernel pressureRhsKernel;
  occa::kernel pressureDirichletBCKernel;

  occa::kernel velocityRhsKernel;
  occa::kernel velocityNeumannBCKernel;
  occa::kernel velocityDirichletBCKernel;

  occa::kernel cflKernel;

  occa::kernel setEllipticCoeffKernel;
  occa::kernel setEllipticCoeffPressureKernel;

  occa::kernel curlKernel;
  occa::kernel maskCopyKernel;
  occa::kernel maskKernel;

  int* EToB;
  int* EToBMesh;
  occa::memory o_EToB;
  occa::memory o_EToBMesh;

  occa::properties *kernelInfo;
};


#include "io.hpp"

// std::to_string might be not accurate enough
static std::string to_string_f(double a)
{
  std::stringstream s;
  s << std::scientific << a;
  return s.str();
}

static std::vector<std::string> serializeString(const std::string sin, char dlim)
{
  std::vector<std::string> slist;
  std::string s(sin);
  s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
  std::stringstream ss;
  ss.str(s);
  while( ss.good() ) {
    std::string substr;
    std::getline(ss, substr, dlim);
    slist.push_back(substr);
  }
  return slist;
}

void evaluateProperties(nrs_t* nrs, const double timeNew);

void compileKernels();

std::vector<int>
determineMGLevels(std::string section);

int numberActiveFields(nrs_t* nrs);

#endif
