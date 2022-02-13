/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "adios2.h"
#include "mgard/compress_cuda.hpp"
#include "postprocessing.hpp"

#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_RESET "\x1b[0m"

using namespace std::chrono;

void print_usage_message(char *argv[], FILE *fp) {
  fprintf(fp,
          "Usage: %s [input file] [num. of dimensions] [1st dim.] [2nd dim.] "
          "[3rd. dim] ... [tolerance] [s]\n",
          argv[0]);
}

int main(int argc, char *argv[]) {

  //    np_size = 150;

  double compress_time = 0.0;
  double decompress_time = 0.0;
  double gpu_compress_time = 0.0;
  double gpu_decompress_time = 0.0;
  double in_time = 0.0;
  double gpu_in_time = 0.0;
  if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    print_usage_message(argv, stdout);
    return 0;
  }

  char *infile; //, *outfile;
  double tol, s = 0;

  int i = 1;
  infile = "../XGC_2/dataset/d3d_coarse_v2_1000.bp/";
  tol = 1e15;
  double job_sz = 1.0;
  printf("Input data: %s ", infile);
  printf("Abs. error bound: %.2e ", tol);
  printf("S: %.2f\n", s);

  adios2::ADIOS ad;
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader = reader_io.Open(infile, adios2::Mode::Read);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  adios2::Engine writer = bpIO.Open("xgc.mgard.bp", adios2::Mode::Write);

  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  if (!var_i_f_in) {
    std::cout << "Didn't find i_f...exit\n";
    exit(1);
  }
  mgard_cuda::SIZE vx = var_i_f_in.Shape()[2];
  mgard_cuda::SIZE vy = var_i_f_in.Shape()[3];
  mgard_cuda::SIZE nnodes = var_i_f_in.Shape()[1];
  mgard_cuda::SIZE local_nnodes = 3;
  mgard_cuda::SIZE nphi = var_i_f_in.Shape()[0];
  mgard_cuda::SIZE offset = 3;
  size_t gb_elements = nphi * vx * local_nnodes * vy;
  size_t local_elements = nphi * vx * local_nnodes * vy;
  size_t lSize = sizeof(double) * gb_elements;
  double *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(double) * local_elements);
  size_t out_size = 0;
    std::vector<mgard_cuda::SIZE> shape = {nphi, local_nnodes, vx, vy};
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
        {0, 3, 0, 0},
        {nphi, local_nnodes, vx, vy}));
    reader.Get<double>(var_i_f_in, in_buff);
    reader.PerformGets();

    double maxv = 0;
    for (size_t i = 0; i < local_elements; i++)
      maxv = (maxv > in_buff[i]) ? maxv : in_buff[i];
    std::cout << "max element: " << maxv << "\n";

    double *mgard_out_buff = NULL;
    //        printf("Start compressing and decompressing with GPU\n");
    mgard_cuda::Array<4, double> in_array(shape);
    in_array.loadData(in_buff);
    //        std::cout << "loadData: " << shape[0] << ", " << shape[1] << ", "
    //        << shape[2] << ", " << shape[3] << "\n";

    mgard_cuda::Handle<4, double> handle(shape);
    //        std::cout << "before compression\n";
    mgard_cuda::Array<1, unsigned char> compressed_array =
        mgard_cuda::compress(handle, in_array, mgard_cuda::error_bound_type::ABS, tol, s);
    //        std::cout << "after compression\n";
    out_size += compressed_array.getShape()[0];

    mgard_cuda::Array<4, double> out_array =
        mgard_cuda::decompress(handle, compressed_array);
    mgard_out_buff = new double[local_elements];
    double* modified_out_buff = new double[local_elements];
    memcpy(mgard_out_buff, out_array.getDataHost(),
           local_elements * sizeof(double));

    double pd_error_b, pd_error_a, density_error_b, density_error_s;
    double upara_error_b, upara_error_a, tperp_error_b, tperp_error_a;
    double tpara_error_b, tpara_error_a;

    std::vector <double> lagranges = compute_lagrange_parameters(infile,
         mgard_out_buff, local_elements, local_nnodes, in_buff, nphi, nnodes,
         vx, vy, offset, modified_out_buff, pd_error_b,
         pd_error_a, density_error_b, density_error_s, upara_error_b,
         upara_error_a, tperp_error_b, tperp_error_a, tpara_error_b,
         tpara_error_a);

    double error_L_inf_norm = 0;
    for (int i = 0; i < local_elements; ++i) {
      double temp = fabs(in_buff[i] - mgard_out_buff[i]);
      if (temp > error_L_inf_norm)
        error_L_inf_norm = temp;
    }
    double absolute_L_inf_error = error_L_inf_norm;

    printf("Abs. L^infty error bound: %10.5E \n", tol);
    printf("Abs. L^infty error: %10.5E \n", absolute_L_inf_error);

    if (absolute_L_inf_error < tol) {
      printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    } else {
      printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
      return -1;
    }

    char write_f[2048];
    FILE *pFile = fopen(write_f, "wb");
    fwrite(mgard_out_buff, sizeof(double), out_size, pFile);
    fclose(pFile);
    delete mgard_out_buff;

    std::cout << " CPU to GPU time: " << gpu_in_time
              << ", compression time: " << gpu_compress_time
              << ", decompress time: " << gpu_decompress_time << "\n";

  mgard_cuda::cudaFreeHostHelper(in_buff);
  size_t gb_compressed;
  reader.Close();

  return 0;
}
