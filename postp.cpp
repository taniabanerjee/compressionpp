#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <adios2.h>
#include <mpi.h>
#include <assert.h>
#include "npy.hpp"

using namespace std;
int ndata = 16395, n_phi = 8, nsize = 39;
void load_npy_file(const char* path, vector <double> &data,
    vector<unsigned long> &shape)
{
    bool fortran_order;
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);
    // for (size_t i = 0; i<data.size(); i++)
        // cout << data[i] << "\n";
    // cout << endl << endl;
    return;
}

void read_mgard_file()
{
    // double* f0_g = (double*)mkl_malloc(n_phi*ndata*sizeof(double), 64);
    std::vector<double> f0_g(8*16395*39*39);
    const char* filename = "/gpfs/alpine/csc143/proj-shared/ljm/MGARD_2/MGARD-SMC/build/v2_1000/uniform/d3d_coarse_v2_1000.bp.mgard.4d.s0.4e15";
    adios2::ADIOS adios;
    adios2::IO io = adios.DeclareIO("CheckpointRestart");
    adios2::Engine reader = io.Open(filename, adios2::Mode::Read);
    adios2::Variable<double> vT = io.InquireVariable<double>("i_f_4d");
    if (vT) {
        reader.Get(vT, f0_g.data());
    }
    reader.Close();

#if 0
 232 for i in range(len(reduced_eb)):
 233     filename = '/gpfs/alpine/csc143/proj-shared/ljm/MGARD_2/MGARD-SMC/build     /v2_{0}/uniform/d3d_coarse_v2_{0}.bp.mgard.4d.s{1}.{2}'.format(timestep,s_v     al,reduced_eb[i])
 234     with ad2.open(filename, 'r') as f:
 235         f0_g = f.read('i_f_4d')
 236     print(f0_g.shape)
 237     print(type(f0_g[0][0][0][0]))
 238     np.save('./results/MGARD_Lagrange_expected/v2_{}/MGARD_uniform_raw/MGAR     D_uniform_{}.npy'.format(timestep, reduced_eb[i]), f0_g)
 239     del f0_g
#endif
}

void qoi_numerator_para(vector <double> &f0_f, vector <double> &vth,
    vector <double> &vp, vector <double> &mu_qoi, vector <double> &vth2,
    double plt_mass, double sml_e_charge)
{
    return;
}


void qoi_numerator(vector <double> &f0_f, vector <double> &vth,
    vector <double> &vp, vector <double> &mu_qoi, vector <double> &vth2,
    double plt_mass, double sml_e_charge)
{
    return;
}

#if 0
void qoi_numerator_matrices(vector <double> &f0_f, vector <double> &vth,
    vector <double> &vp, vector <double> &mu_qoi, vector <double> &vth2,
    double plt_mass, double sml_e_charge)
{
    vector <double> den;
    vector <double> s_den;
    vector <double> V2;
    int start = plane*ndata*nsize*nsize;

    for (i=0; i<ndata*nsize*nsize; ++i) {
        den.push_back(f0_f[i+start] * vol[i]);
    }

    for (i=0; i<ndata; ++i) {
        double value = 0
        for (j=0; j<nsize*nsize; ++j) {
            value += den[j];
        }
        s_den.push_back(value);
    }
    for (i=0; i<ndata; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V2.push_back(vol[nsize*nsize*i + nsize*j + k] *
                    vth[i]*vp[k]);
    }

    return V2;
    return;
}
#endif

vector <double> qoi_V2(vector <double> &vol, vector <double> &vth,
    vector <double> &vp)
{
    int i, j, k;
    vector <double> V2;

    for (i=0; i<ndata; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V2.push_back(vol[nsize*nsize*i + nsize*j + k] *
                    vth[i]*vp[k]);
    }

    return V2;
}

vector <double> qoi_V3(vector <double> &vol, vector <double> &vth2,
    vector <double> &mu_qoi, double ptl_mass)
{
    int i, j, k;
    vector <double> V3;

    for (i=0; i<ndata; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V3.push_back(vol[nsize*nsize*i + nsize*j + k] * 0.5 *
                    mu_qoi[k] * vth2[i] * ptl_mass);
    }

    return V3;
}

vector <double> qoi_V4(vector <double> &vol, vector <double> &vth2,
    vector <double> &vp, double ptl_mass)
{
    int i, j, k;
    vector <double> V4;

    for (i=0; i<ndata; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V4.push_back(vol[nsize*nsize*i + nsize*j + k] *
                    vp[k] * vth2[i] * ptl_mass);
    }

    return V4;
}

void checkVectorsV(vector <double> V2, vector <double> V3, vector <double> V4, vector <double> Tpara)
{
    vector<unsigned long> V2_data_shape;
    vector <double> V2_data;
    load_npy_file("./V2.npy", V2_data, V2_data_shape);
    int p, i;
    for (i=0; i<V2_data.size(); ++i) {
        if (abs(V2_data[i]-V2[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V2_data[i], V2[i]);
        }
    }
    vector<unsigned long> V3_data_shape;
    vector <double> V3_data;
    load_npy_file("./V3.npy", V3_data, V3_data_shape);
    for (i=0; i<V3_data.size(); ++i) {
        if (abs(V3_data[i]-V3[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V3_data[i], V3[i]);
        }
    }
    vector<unsigned long> V4_data_shape;
    vector <double> V4_data;
    load_npy_file("./V4.npy", V4_data, V4_data_shape);
    for (i=0; i<V4_data.size(); ++i) {
        if (abs(V4_data[i]-V4[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V4_data[i], V4[i]);
        }
    }
}

double rmse_error(vector <double> &x, vector <double> &y)
{
    unsigned int xsize = x.size();
    unsigned int ysize = y.size();
    assert(xsize == ysize);
    double e;
    for (int i=0; i<xsize; ++i) {
        e += pow((x[i] - y[i]), 2);
    }
    return sqrt(e/xsize);
}

bool isConverged(vector <double> difflist)
{
    bool status = true;
    unsigned int vsize = difflist.size();
    double firstVal = difflist[0];
    double lastVal = difflist[vsize-1];
    if ((lastVal - firstVal) > 0) {
        status = false;
    }
    return status;
}

int main(int argc, char** argv)
{
#if 0
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

    printf("hello world\n");
    // read_mgard_file();

    // get the MGARD results
    // Remove non-negative values
    //const char* path = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_raw/MGARD_uniform_4e15.npy";
    const char* path = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_4e15_nonngegative_relu.npy";
    vector<unsigned long> mgard_data_shape;
    vector <double> recon;
    load_npy_file(path, recon, mgard_data_shape);

    const char* ipath = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_raw/MGARD_uniform_4e15.npy";
    vector<unsigned long> i_data_shape;
    vector <double> i_f;
    load_npy_file(ipath, i_f, i_data_shape);

    // get the output of xgcexp.f0_diag_vol_origdim
    double sml_e_charge = 1.6022e-19;
    double ptl_mass = 3.344e-27;
    vector<unsigned long> vol_data_shape;
    vector <double> vol;
    load_npy_file("./vol.npy", vol, vol_data_shape);
    vector<unsigned long> vth_data_shape;
    vector <double> vth;
    load_npy_file("./vth.npy", vth, vth_data_shape);
    vector<unsigned long> vp_data_shape;
    vector <double> vp;
    load_npy_file("./vp.npy", vp, vp_data_shape);
    vector<unsigned long> mu_qoi_data_shape;
    vector <double> mu_qoi;
    load_npy_file("./mu_qoi.npy", mu_qoi, mu_qoi_data_shape);
    vector<unsigned long> vth2_data_shape;
    vector <double> vth2;
    load_npy_file("./vth2.npy", vth2, vth2_data_shape);
    vector<unsigned long> f0_grid_vol_data_shape;
    vector <double> f0_grid_vol;
    load_npy_file("./f0_grid_vol.npy", f0_grid_vol, f0_grid_vol_data_shape);
    vector<unsigned long> mu_vp_vol_data_shape;
    vector <double> mu_vp_vol;
    load_npy_file("./mu_vp_vol.npy", mu_vp_vol, mu_vp_vol_data_shape);
    // get the actual QoIs
    vector<unsigned long> den_f_data_shape;
    vector <double> den_f;
    load_npy_file("./den_f.npy", den_f, den_f_data_shape);
    vector<unsigned long> upara_f_data_shape;
    vector <double> upara_f;
    load_npy_file("./upara_f.npy", upara_f, upara_f_data_shape);
    vector<unsigned long> tperp_f_data_shape;
    vector <double> tperp_f;
    load_npy_file("./tperp_f.npy", tperp_f, tperp_f_data_shape);
    vector<unsigned long> tpara_f_data_shape;
    vector <double> tpara_f;
    load_npy_file("./tpara_f.npy", tpara_f, tpara_f_data_shape);

    // Assign recon_breg, lagranges
    int count = 0;
    // double* recon_breg = (double*) malloc(n_phi*ndata*nsize*nsize);
    // double* lagranges = (double*) malloc(n_phi*ndata*4);
    vector <double> recon_breg;
    vector <double> lagranges;
    double gradients[4] = {0.0, 0.0, 0.0, 0.0};
    double hessians[4][4] = {0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0};
    double K[nsize*nsize];
    double breg_result[nsize*nsize];
    memset(K, 0, nsize*nsize*sizeof(double));
    vector <double> V2 = qoi_V2(vol, vth, vp);
    vector <double> V3 = qoi_V3(vol, vth2, mu_qoi, ptl_mass);
    vector <double> V4 = qoi_V4(vol, vth2, vp, ptl_mass);
    vector <double> tpara_data;
    int p, i, idx;
    for (p=0; p<8; ++p) {
        for (i=0; i<ndata; ++i) {
            tpara_data.push_back(sml_e_charge * tpara_f[ndata*p + i] +
                vth2[i] * ptl_mass * pow((upara_f[ndata*p + i]/vth[i]), 2));
        }
    }
    // compare vectors V2, V3, V4, and Tpara
    checkVectorsV(V2, V3, V4, tpara_data);
    for (p=0; p<8; ++p) {
        for (idx=0; idx<ndata; ++idx) {
            double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
            double D = den_f[p*ndata + idx];
            double U = upara_f[p*ndata + idx];
            double Tperp = tperp_f[p*ndata + idx];
            double Tpara = tpara_data[p*ndata + idx];
            vector <double> L2_den;
            vector <double> L2_upara;
            vector <double> L2_tperp;
            vector <double> L2_tpara;
            while (1) {
                for (i=0; i<nsize*nsize; ++i) {
                    K[i] = lambdas[0]*vol[nsize*nsize*idx + i] +
                           lambdas[1]*V2[nsize*nsize*idx + i] +
                           lambdas[2]*V3[nsize*nsize*idx + i] +
                           lambdas[3]*V4[nsize*nsize*idx + i];
                }
                double update_D, update_U, update_Tperp, update_Tpara;
                if (count > 0) {
                    for (i=0; i<nsize*nsize; ++i) {
                        breg_result[i] = recon[nsize*nsize*idx + i]*
                            exp(-K[i]);
                        update_D += breg_result[i]*vol[
                            nsize*nsize*idx + i];
                        update_U += breg_result[i]*V2[
                            nsize*nsize*idx + i];
                        update_Tperp += breg_result[i]*V3[
                            nsize*nsize*idx + i];
                        update_Tpara += breg_result[i]*V4[
                            nsize*nsize*idx + i];
                    }
                    L2_den.push_back(pow((update_D-D), 2));
                    L2_upara.push_back(pow((update_U-U), 2));
                    L2_tperp.push_back(pow((update_Tperp-Tperp), 2));
                    L2_tpara.push_back(pow((update_Tpara-Tpara), 2));
                    bool converged = (isConverged(L2_den) &&
                           isConverged(L2_upara) && isConverged(
                           L2_tpara) && isConverged(L2_tperp));
                    if (converged) {
                        for (i=0; i<nsize*nsize; ++i) {
                            recon_breg.push_back(breg_result[i]);
                        }
                        lagranges.push_back(lambdas[0]);
                        lagranges.push_back(lambdas[1]);
                        lagranges.push_back(lambdas[2]);
                        lagranges.push_back(lambdas[3]);
                    }
                    else if (count == 20 && !converged){
                        printf ("Node did not converge\n");
                    }
                }
                double gvalue1 = 0, gvalue2 = 0, gvalue3 = 0, gvalue4 = 0;
                double hvalue1 = 0, hvalue2 = 0, hvalue3 = 0, hvalue4 = 0;
                double hvalue5 = 0, hvalue6 = 0, hvalue7 = 0;
                double hvalue8 = 0, hvalue9 = 0, hvalue10 = 0;

                for (i=0; i<nsize*nsize; ++i) {
                    gvalue1 += recon[nsize*nsize*idx + i]*
                      vol[nsize*nsize*idx + i]*exp(-K[i]) +
                      den_f[p*ndata + idx];
                    gvalue2 += recon[nsize*nsize*idx + i]*
                      V2[nsize*nsize*idx + i]*exp(-K[i]) +
                      upara_f[p*ndata + idx]*den_f[p*ndata + idx];
                    gvalue3 += recon[nsize*nsize*idx + i]*
                      V3[nsize*nsize*idx + i]*exp(-K[i]) +
                      tperp_f[p*ndata + idx]*den_f[p*ndata + idx]*sml_e_charge;
                    gvalue4 += recon[nsize*nsize*idx + i]*
                      V4[nsize*nsize*idx + i]*exp(-K[i]) +
                      tpara_f[p*ndata + idx]*den_f[p*ndata + idx];
                    hvalue1 += recon[nsize*nsize*idx + i]*pow(
                        vol[nsize*nsize*idx + i], 2)*exp(-K[i]);
                    hvalue2 += recon[nsize*nsize*idx + i]* vol[
                        nsize*nsize*idx + i]*V2[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue3 += recon[nsize*nsize*idx + i]* vol[
                        nsize*nsize*idx + i]*V3[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue4 += recon[nsize*nsize*idx + i]* vol[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue5 += recon[nsize*nsize*idx + i]*pow(
                        V2[nsize*nsize*idx + i],2)*exp(-K[i]);
                    hvalue6 += recon[nsize*nsize*idx + i]*V2[
                        nsize*nsize*idx + i]*V3[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue7 += recon[nsize*nsize*idx + i]*V2[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue8 += recon[nsize*nsize*idx + i]*pow(
                        V3[nsize*nsize*idx + i],2)*exp(-K[i]);
                    hvalue9 += recon[nsize*nsize*idx + i]*V3[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue10 += recon[nsize*nsize*idx + i]*pow(
                        V4[nsize*nsize*idx + i],2)*exp(-K[i]);
                }
                gradients[0] = gvalue1;
                gradients[1] = gvalue2;
                gradients[2] = gvalue3;
                gradients[3] = gvalue4;
                hessians[0][0] = hvalue1;
                hessians[0][1] = hvalue2;
                hessians[0][2] = hvalue3;
                hessians[0][3] = hvalue4;
                hessians[1][0] = hvalue1;
                hessians[1][1] = hvalue5;
                hessians[1][2] = hvalue6;
                hessians[1][3] = hvalue7;
                hessians[2][0] = hvalue3;
                hessians[2][1] = hvalue6;
                hessians[2][2] = hvalue8;
                hessians[2][3] = hvalue9;
                hessians[3][0] = hvalue4;
                hessians[3][1] = hvalue7;
                hessians[3][2] = hvalue9;
                hessians[3][3] = hvalue10;
                // compute lambdas
                count = count + 1;
            }
        }
    }
    //compute reconstruction RMSE
    double recon_rmse = rmse_error(i_f, recon);
    //compute breg RMSE
    double breg_rmse = rmse_error(i_f, recon_breg);
    printf ("recon PD rmse: %g, breg PD rmse: %g\n", recon_rmse, breg_rmse);

#if 0
    // compute compression ratio using 1 plane
    double data_size = ndata * nsize * nsize * 64;
    double lagrange_cost = ndata * 4 * 64;
    double compression_ratio = data_size/((data_size/
          mgard_compression_ratio) + lagrange_cost);

    printf ("compression_ratio_fixed_cost_MGARD_breg: %g\n",
        compression_ratio);

    // store Lagranges with PQ
#endif
}
