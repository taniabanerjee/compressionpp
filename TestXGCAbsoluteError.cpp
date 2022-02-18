/*
 * Copyright 2021, Oak Ridge National Laboratory.
 * MGARD-GPU: MultiGrid Adaptive Reduction of Data Accelerated by GPUs
 * Author: Jieyang Chen (chenj3@ornl.gov)
 * Date: April 2, 2021
 */

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

  MPI_Init(&argc, &argv);
  int rank, np_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np_size);
  //    np_size = 150;
  /*
      int deviceCount;
      cudaGetDeviceCount(&deviceCount);               // How many GPUs?
      int device_id = rank % deviceCount;
      cudaSetDevice(device_id);
      std::cout << "total number of devices: " << deviceCount << ", rank " <<
     rank << " used " << device_id << "\n";
  */

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
  double tol, s = 0, bigtest = 0;
  int timestep=0;

  int i = 1;
  infile = argv[i++];
  tol = atof(argv[i++]);
  s = atof(argv[i++]);
  double job_sz = atof(argv[i++]);
  bigtest = atof(argv[i++]);
  timestep = atof(argv[i++]);
  if (rank == 0) {
    printf("Input data: %s ", infile);
    printf("Abs. error bound: %.2e ", tol);
    printf("S: %.2f\n", s);
  }

  adios2::ADIOS ad("", MPI_COMM_WORLD);
  adios2::IO reader_io = ad.DeclareIO("XGC");
  adios2::Engine reader = reader_io.Open(infile, adios2::Mode::Read);
  adios2::IO bpIO = ad.DeclareIO("WriteBP_File");
  // adios2::Engine writer = bpIO.Open("xgc_compressed.mgard.bp", adios2::Mode::Write);
  // adios2::Engine writer_lag = bpIO.Open("xgc_lagrange.mgard.bp", adios2::Mode::Write);

  adios2::Variable<double> var_i_f_in;
  var_i_f_in = reader_io.InquireVariable<double>("i_f");
  if (!var_i_f_in) {
    std::cout << "Didn't find i_f...exit\n";
    exit(1);
  }
  int vxIndex = 2;
  int vyIndex = 3;
  int nodeIndex = 1;
  int planeIndex = 0;
  if (bigtest) {
      vxIndex = 1;
      vyIndex = 3;
      nodeIndex = 2;
      planeIndex = 0;
  }
  mgard_cuda::SIZE vx = var_i_f_in.Shape()[vxIndex];
  mgard_cuda::SIZE vy = var_i_f_in.Shape()[vyIndex];
  mgard_cuda::SIZE nnodes = var_i_f_in.Shape()[nodeIndex];
  mgard_cuda::SIZE nphi = var_i_f_in.Shape()[planeIndex];
  size_t gb_elements = nphi * vx * nnodes * vy;
  size_t num_iter =
      (size_t)(std::ceil)((double)gb_elements * sizeof(double) / 1024.0 /
                          1024.0 / 1024.0 / job_sz / np_size);
  size_t div_nnodes = (size_t)(std::ceil)((double)nnodes / num_iter);
  size_t iter_nnodes =
      (size_t)(std::ceil)((double)div_nnodes / (double)np_size);
  //    size_t iter_elements = iter_nnodes * vx * vy * nphi;
  mgard_cuda::SIZE local_nnodes =
      (rank == np_size - 1) ? (div_nnodes - rank * iter_nnodes) : iter_nnodes;
  size_t local_elements = nphi * vx * local_nnodes * vy;
  size_t lSize = sizeof(double) * gb_elements;
  double *in_buff;
  mgard_cuda::cudaMallocHostHelper((void **)&in_buff,
                                   sizeof(double) * local_elements);
  if (rank == 0) {
    std::cout << "total data size: {" << nphi << ", " << nnodes << ", "
      << vx << ", " << vy << "}, number of iters: " << num_iter << "\n";
  }
  size_t out_size = 0;
  size_t lagrange_size = 0;
  for (size_t iter = 0; iter < num_iter; iter++) {
    if (iter == num_iter - 1) {
      iter_nnodes = (size_t)(std::ceil)(
          ((double)(nnodes - div_nnodes * iter)) /
          (double)np_size); // local_nnodes - iter_nnodes*iter;
      local_nnodes =
          (rank == np_size - 1)
              ? (nnodes - div_nnodes * iter - iter_nnodes * (np_size - 1))
              : iter_nnodes;
      local_elements = local_nnodes * vx * vy * nphi;
    }
    std::vector<mgard_cuda::SIZE> shape = {nphi, local_nnodes, vx, vy};
    if (bigtest) {
        shape[1] = vx;
        shape[2] = local_nnodes;
    }
    long unsigned int offset = div_nnodes * iter + iter_nnodes * rank;
    // adios2::Variable<double> bp_ldata = bpIO.DefineVariable<double>(
      // "lag_p", {nphi, nnodes}, {0, offset}, {nphi, local_nnodes});
    if (bigtest) {
        std::cout << "rank " << rank << " read from {0, 0, "
              << offset << ", 0} for {" << nphi << ", " << vx << ", "
              << local_nnodes << ", " << vy << "}\n";
    }
    else {
        std::cout << "rank " << rank << " read from {0, "
              << offset << ", 0, 0} for {" << nphi
              << ", " << local_nnodes << ", " << vx<< ", " << vy << "}\n";
    }
    std::vector<unsigned long> dim1 = {0, offset, 0, 0};
    std::vector<unsigned long> dim2 = {nphi, local_nnodes, vx, vy};
    std::pair<std::vector<unsigned long>, std::vector<unsigned long>> dim;
    dim.first = dim1;
    dim.second = dim2;
    if (bigtest) {
        dim1[1] = 0;
        dim1[2] = offset;
        dim2[1] = vx;
        dim2[2] = local_nnodes;
    }

    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(dim));
    reader.Get<double>(var_i_f_in, in_buff);
    reader.PerformGets();

    double maxv = 0;
    for (size_t i = 0; i < local_elements; i++)
      maxv = (maxv > in_buff[i]) ? maxv : in_buff[i];
    std::cout << "max element: " << maxv << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    double *mgard_out_buff = NULL;
    //        printf("Start compressing and decompressing with GPU\n");
    mgard_cuda::Array<4, double> in_array(shape);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      in_time = -MPI_Wtime();
    }
    in_array.loadData(in_buff);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_in_time += (in_time + MPI_Wtime());
    }
    //        std::cout << "loadData: " << shape[0] << ", " << shape[1] << ", "
    //        << shape[2] << ", " << shape[3] << "\n";

    mgard_cuda::Handle<4, double> handle(shape);
    //        std::cout << "before compression\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      compress_time = -MPI_Wtime();
    }
    mgard_cuda::Array<1, unsigned char> compressed_array =
        mgard_cuda::compress(handle, in_array, mgard_cuda::error_bound_type::ABS, tol, s);
    //        std::cout << "after compression\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_compress_time += (compress_time + MPI_Wtime());
    }
    out_size += compressed_array.getShape()[0];

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      decompress_time = -MPI_Wtime();
    }
    mgard_cuda::Array<4, double> out_array =
        mgard_cuda::decompress(handle, compressed_array);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      gpu_decompress_time += (decompress_time + MPI_Wtime());
    }
    mgard_out_buff = new double[local_elements];
    double* modified_out_buff = new double[local_elements];
    memcpy(mgard_out_buff, out_array.getDataHost(),
           local_elements * sizeof(double));

    vector <const char*> error_names = {"PD error", "Density error", "Upara error", "Tperp error", "Tpara error", "n0 error", "T0 error"};
    double* before_errors = new double [error_names.size()];
    double* after_errors = new double [error_names.size()];

    vector <double> lagranges = compute_lagrange_parameters(infile,
         mgard_out_buff, local_elements, local_nnodes, in_buff, nphi,
         nnodes, vx, vy, offset, maxv, modified_out_buff,
         before_errors, after_errors);

    lagrange_size += nphi * local_nnodes * 4;
    // double error_L_inf_norm = 0;
    // for (int i = 0; i < local_elements; ++i) {
    //   double temp = fabs(in_buff[i] - modified_out_buff[i]);
    //   if (temp > error_L_inf_norm)
    //     error_L_inf_norm = temp;
    // }
    // double absolute_L_inf_error = error_L_inf_norm;

    double error_L2_norm = 0;
    for (int i = 0; i < local_elements; ++i) {
        double temp = pow((in_buff[i] - modified_out_buff[i]),2);
        error_L2_norm += temp;
    }
    double L2_error = sqrt(error_L2_norm/local_elements);
    printf("L2 error bound: %10.5E \n", tol);
    printf("L2 error: %10.5E \n", L2_error);

    // if (absolute_L_inf_error < tol) {
    //   printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    // } else {
    //   printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
    //   // return -1;
    // }
    if (L2_error < tol) {
      printf(ANSI_GREEN "SUCCESS: Error tolerance met!" ANSI_RESET "\n");
    } else {
      printf(ANSI_RED "FAILURE: Error tolerance NOT met!" ANSI_RESET "\n");
      // return -1;
    }
    char write_f[2048];
    sprintf(write_f, "xgc.mgard.rank%i_%zu_%i.bin", rank, iter, timestep);
    FILE *pFile = fopen(write_f, "wb");
    fwrite(mgard_out_buff, sizeof(double), out_size, pFile);
    fclose(pFile);
    sprintf(write_f, "xgc.qoi.rank%i_%zu_%i.txt", rank, iter, timestep);
    FILE* pFileTxt = fopen(write_f, "w");
    char output[100];
    for (int fi=0; fi < error_names.size(); ++fi) {
        sprintf(output, "%s, %5.3g, %5.3g\n", error_names[fi],
            before_errors[fi], after_errors[fi]);
        fputs(output, pFileTxt);
    }
    fclose(pFileTxt);
/*
    // MPI_Barrier(MPI_COMM_WORLD);
    unsigned char *mgard_compress_buff = new unsigned char[out_size];
    memcpy(mgard_compress_buff, compressed_array.getDataHost(), out_size);
    bp_ldata.SetSelection(adios2::Box<adios2::Dims>(
          {0, offset},
          {nphi, local_nnodes}));
    writer_lag.Put<double>(bp_ldata, lagranges.data());
    writer_lag.PerformPuts();
*/
    delete mgard_out_buff;
  }
  // writer_lag.Close();
  if (rank == 0) {
    std::cout << " CPU to GPU time: " << gpu_in_time
              << ", compression time: " << gpu_compress_time
              << ", decompress time: " << gpu_decompress_time << "\n";
  }

  mgard_cuda::cudaFreeHostHelper(in_buff);
  size_t gb_compressed, gb_compressed_lag;
  MPI_Allreduce(&out_size, &gb_compressed, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&lagrange_size, &gb_compressed_lag, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    printf("In size:  %10ld  Out size: %10ld  Lagrange size: %10ld  Compression ratio: %f \n", lSize,
           gb_compressed, gb_compressed_lag, (double)lSize / (gb_compressed + gb_compressed_lag));
  }
  reader.Close();

  //    size_t exscan;
  //    size_t *scan_counts = (size_t *)malloc(np_size * sizeof(size_t));
  //    MPI_Exscan(&out_size, &exscan, 1, MPI_UNSIGNED_LONG, MPI_SUM,
  //    MPI_COMM_WORLD); std::cout << "rank " << rank << " compressed size: " <<
  //    out_size << ", exscan: " << exscan << ", total compressed: " <<
  //    gb_compressed << "\n"; MPI_Gather(&exscan, 1, MPI_UNSIGNED_LONG,
  //    scan_counts, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD); if (rank == 0) {
  //        scan_counts[0] = 0;
  //        exscan = 0;
  //        std::cout << "scanned counts: ";
  //        for (int i=0; i<np_size; i ++)
  //            std::cout << scan_counts[i] << ", ";
  //        std::cout << "\n";
  //    }
  //    unsigned char *mgard_compress_buff = new unsigned char[out_size];
  //    memcpy(mgard_compress_buff, compressed_array.getDataHost(), out_size);

  //    adios2::Variable<unsigned char> bp_fdata = bpIO.DefineVariable<unsigned
  //    char>(
  //      "mgard_f", {gb_compressed}, {exscan}, {out_size},
  //      adios2::ConstantDims);
  //    writer.Put<unsigned char>(bp_fdata, mgard_compress_buff);
  //    if (rank==0) {
  //        adios2::Variable<size_t> bp_count = bpIO.DefineVariable<size_t>(
  //        "block_start", {(size_t)np_size}, {0}, {(size_t)np_size},
  //        adios2::ConstantDims); writer.Put<size_t>(bp_count, scan_counts);
  //    }
  //    writer.Close();

  MPI_Finalize();

  return 0;
}
