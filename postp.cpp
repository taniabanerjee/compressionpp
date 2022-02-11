#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <adios2.h>
#include <mpi.h>
#include <assert.h>
#include <time.h>
#include "npy.hpp"

using namespace std;
double determinant(double a[4][4], double k);
double** cofactor(double num[4][4], double f);
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

void read_xgc_file(const char* filepath, vector <double> &i_f, int &nnodes, int &nphi, int &nsize)
{
    adios2::ADIOS ad;
    adios2::IO reader_io = ad.DeclareIO("XGC");
    adios2::Engine reader = reader_io.Open(filepath, adios2::Mode::Read);
    // Inquire variable
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = reader_io.InquireVariable<double>("i_f");
    std::vector<std::size_t> shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>(
        {0, 0, 0, 0}, {shape[0], shape[1], shape[2], shape[3]}));
    reader.Get<double>(var_i_f_in, i_f);
    reader.Close();
    nphi = shape[0];
    nnodes = shape[1];
    nsize = shape[2];
    size_t num_elements = shape[0] * shape[1] * shape[2] * shape[3];
    printf("Read in: %s\n", filepath);
    printf(" XGC data shape: (%ld, %ld, %ld, %ld)\n ", shape[0], shape[1],
           shape[2], shape[3]);
    return;
}

void set_vol(vector <double> &grid_vol,
    double f0_nvp, double f0_nmu, int nnodes, vector <double> &vol)
{
    std::vector<double> vp_vol;
    vp_vol.push_back(0.5);
    for (int ii = 1; ii<f0_nvp*2; ++ii) {
        vp_vol.push_back(1.0);
    }
    vp_vol.push_back(0.5);

    std::vector<double> mu_vol;
    mu_vol.push_back(0.5);
    for (int ii = 1; ii<f0_nmu; ++ii) {
        mu_vol.push_back(1.0);
    }
    mu_vol.push_back(0.5);

    std::vector<double> mu_vp_vol;
    for (int ii=0; ii<mu_vol.size(); ++ii) {
        for (int jj=0; jj<mu_vol.size(); ++jj) {
            mu_vp_vol.push_back(mu_vol[ii] * vp_vol[jj]);
        }
    }
    for (int ii=0; ii<nnodes; ++ii) {
        for (int jj=0; jj<mu_vp_vol.size(); ++jj) {
        // Access grid_vol with an offset of nnodes to get to the electrons
            vol.push_back(grid_vol[nnodes+ii] * mu_vp_vol[jj]);
        }
    }
    return;
}

void set_vp(double f0_nvp, double f0_dvp, vector <double> &vp)
{
    for (int ii = -f0_nvp; ii<f0_nvp+1; ++ii) {
        vp.push_back(ii*f0_dvp);
    }
    return;
}

void set_mu_qoi(double f0_nmu, double f0_dsmu, vector <double> &mu_qoi)
{
    for (int ii = 0; ii<f0_nmu+1; ++ii) {
        mu_qoi.push_back(pow(ii*f0_dsmu, 2));
    }
    return;
}

void set_vth2(vector <double> &f0_T_ev, double nnodes, double sml_e_charge, double ptl_mass, vector <double> &vth2, vector <double> &vth)
{
    double value = 0;
    for (int ii=0; ii<nnodes; ++ii) {
        // Access f0_T_ev with an offset of nnodes to get to the electrons
        value = f0_T_ev[nnodes+ii]*sml_e_charge/ptl_mass;
        vth2.push_back(value);
        vth.push_back(sqrt(value));
    }
    return;
}

void read_f0_params(const char* datapath, vector <double> i_f,
    int nnodes, int nphi, int nsize, vector <double> &vol,
    vector <double> &vth, vector <double> &vp, vector <double> &mu_qoi,
    vector <double> &vth2, double &sml_e_charge, double &ptl_mass)
{
    sml_e_charge = 1.6022e-19;
    ptl_mass = 3.344e-27;

    adios2::ADIOS ad;
    adios2::IO read_vol_io = ad.DeclareIO("xgc_vol");
    char vol_file[2048];
    sprintf(vol_file, "%sxgc.f0.mesh.bp", datapath);
    adios2::Engine reader_vol = read_vol_io.Open(vol_file, adios2::Mode::Read);

    adios2::Variable<double> var_i_f_in;
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    vector<std::size_t> vol_shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {vol_shape[0], vol_shape[1]}));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);

    adios2::Variable<int> var_i_f_nvp;
    var_i_f_nvp = read_vol_io.InquireVariable<int>("f0_nvp");
    std::vector<int> f0_nvp;
    reader_vol.Get<int>(var_i_f_nvp, f0_nvp);

    adios2::Variable<int> var_i_f_nmu;
    var_i_f_nvp = read_vol_io.InquireVariable<int>("f0_nmu");
    std::vector<int> f0_nmu;
    reader_vol.Get<int>(var_i_f_nvp, f0_nmu);

    adios2::Variable<double> var_i_f_dvp;
    var_i_f_dvp = read_vol_io.InquireVariable<double>("f0_dvp");
    std::vector<double> f0_dvp;
    reader_vol.Get<double>(var_i_f_dvp, f0_dvp);

    var_i_f_in = read_vol_io.InquireVariable<double>("f0_dsmu");
    std::vector<double> f0_dsmu;
    reader_vol.Get<double>(var_i_f_in, f0_dsmu);

    var_i_f_in = read_vol_io.InquireVariable<double>("f0_T_ev");
    unsigned long num_nodes = var_i_f_in.Shape()[1];
    vector<std::size_t> ev_shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {ev_shape[0], ev_shape[1]}));
    std::vector<double> f0_T_ev;
    reader_vol.Get<double>(var_i_f_in, f0_T_ev);
    reader_vol.Close();

    set_vol(grid_vol, f0_nvp[0], f0_nmu[0], nnodes, vol);
    set_vp(f0_nvp[0], f0_dvp[0], vp);
    set_mu_qoi(f0_nmu[0], f0_dsmu[0], mu_qoi);
    set_vth2(f0_T_ev, nnodes, sml_e_charge, ptl_mass, vth2, vth);
}

void read_mgard_file()
{
    // double* f0_g = (double*)mkl_malloc(n_phi*nnodes*sizeof(double), 64);
    // std::vector<double> f0_g(8*16395*39*39);
#if 0
    const char* datapath = "../XGC_2/dataset/";
    adios2::IO read_vol_io = ad.DeclareIO("xgc_vol");
    char vol_file[2048];
    sprintf(vol_file, "%sxgc.mesh.bp", datapath);
    adios2::Engine reader_vol = read_vol_io.Open(vol_file, adios2::Mode::Read);
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    size_t num_nodes = var_i_f_in.Shape()[1];
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, num_nodes}));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);
    reader_vol.Close();
#endif
    return;
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

vector <double> qoi_V2(vector <double> &vol, vector <double> &vth,
    vector <double> &vp, int nnodes, int nsize)
{
    int i, j, k;
    vector <double> V2;

    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V2.push_back(vol[nsize*nsize*i + nsize*j + k] *
                    vth[i]*vp[k]);
    }

    return V2;
}

vector <double> qoi_V3(vector <double> &vol, vector <double> &vth2,
    vector <double> &mu_qoi, double ptl_mass, int nnodes, int nsize)
{
    int i, j, k;
    vector <double> V3;

    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V3.push_back(vol[nsize*nsize*i + nsize*j + k] * 0.5 *
                    mu_qoi[j] * vth2[i] * ptl_mass);
    }

    return V3;
}

vector <double> qoi_V4(vector <double> &vol, vector <double> &vth2,
    vector <double> &vp, double ptl_mass, int nnodes, int nsize)
{
    int i, j, k;
    vector <double> V4;

    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                V4.push_back(vol[nsize*nsize*i + nsize*j + k] *
                    vp[k]*vp[k] * vth2[i] * ptl_mass);
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
        // printf("i = %d, data-read = %g, data-computed= %g\n", i, V4_data[i], V4[i]);
    }
    vector<unsigned long> Tpara_data_shape;
    vector <double> Tpara_data;
    load_npy_file("./Tpara.npy", Tpara_data, Tpara_data_shape);
    for (i=0; i<Tpara_data.size(); ++i) {
        if (abs(Tpara_data[i]-Tpara[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, Tpara_data[i], Tpara[i]);
        }
        // printf("i = %d, data-read = %g, data-computed= %g\n", i, Tpara_data[i], Tpara[i]);
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

bool isConverged(vector <double> difflist, double eB)
{
    bool status = false;
    unsigned int vsize = difflist.size();
    if (vsize < 2) {
         return status;
    }
    double last2Val = difflist[vsize-2];
    double last1Val = difflist[vsize-1];
    if (abs(last2Val - last1Val) < eB) {
        status = true;
    }
    return status;
}

void compute_C_qois(vector <double> i_f, int iphi, int nnodes, int nsize,
    vector <double> &vol, vector <double> &vth,
    vector <double> &vp, vector <double> &mu_qoi, vector <double> &vth2,
    double ptl_mass, double sml_e_charge, vector <double> &den_f,
    vector <double> &upara_f, vector <double> &tperp_f,
    vector <double> &tpara_f, vector <double> den_f_data, vector <double> upara_f_data, vector <double> tperp_f_data, vector <double> tpara_f_data)
{
    vector <double> den;
    vector <double> upar;
    vector <double> upar_;
    vector <double> tper;
    vector <double> en;
    vector <double> T_par;
    int i, j, k;
    double* f0_f = &i_f[iphi*nnodes*nsize*nsize];
    int den_index = iphi*nnodes;

    for (i=0; i<nnodes*nsize*nsize; ++i) {
        den.push_back(f0_f[i] * vol[i]);
    }

    double value = 0;
    for (i=0; i<nnodes; ++i) {
        value = 0;
        for (j=0; j<nsize*nsize; ++j) {
            value += den[nsize*nsize*i + j];
        }
        den_f.push_back(value);
    }
    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                upar.push_back(f0_f[nsize*nsize*i + nsize*j + k] *
                    vol[nsize*nsize*i + nsize*j + k] *
                    vth[i]*vp[k]);
    }
    for (i=0; i<nnodes; ++i) {
        double value = 0;
        for (j=0; j<nsize*nsize; ++j) {
            value += upar[nsize*nsize*i + j];
        }
        upara_f.push_back(value/den_f[den_index + i]);
    }
    for (i=0; i<nnodes; ++i) {
        upar_.push_back(upara_f[den_index + i]/vth[i]);
    }
    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                tper.push_back(f0_f[nsize*nsize*i + nsize*j + k] *
                    vol[nsize*nsize*i + nsize*j + k] * 0.5 *
                    mu_qoi[j] * vth2[i] * ptl_mass);
    }
    for (i=0; i<nnodes; ++i) {
        double value = 0;
        for (j=0; j<nsize*nsize; ++j) {
            value += tper[nsize*nsize*i + j];
        }
        tperp_f.push_back(value/den_f[den_index + i]/sml_e_charge);
    }
    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            en.push_back(0.5*pow((vp[j]-upar_[i]),2));
    }
    for (i=0; i<nnodes; ++i) {
        for (j=0; j<nsize; ++j)
            for (k=0; k<nsize; ++k)
                T_par.push_back(f0_f[nsize*nsize*i + nsize*j + k] *
                    vol[nsize*nsize*i + nsize*j + k] *
                    en[nsize*i+k] * vth2[i] * ptl_mass);
    }
    for (i=0; i<nnodes; ++i) {
        double value = 0;
        for (j=0; j<nsize*nsize; ++j) {
            value += T_par[nsize*nsize*i + j];
        }
        tpara_f.push_back(2.0*value/den_f[den_index + i]/sml_e_charge);
    }
}

int main(int argc, char** argv)
{
    int nnodes, nphi, nsize;
    clock_t start, end;
    double cpu_time_used;
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

    // read_mgard_file();

    // get the MGARD results
    // Remove non-negative values
    const char* path = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_raw/MGARD_uniform_4e15.npy";
    // const char* path = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_4e15_nonngegative_relu.npy";
    vector<unsigned long> mgard_data_shape;
    vector <double> recon;
    load_npy_file(path, recon, mgard_data_shape);

    start = clock();
    for (int ii=0; ii<recon.size(); ++ii)
    {
        if (!(recon[ii] > 0)) {
            recon[ii] = 100;
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC;
    printf ("%g seconds\n", cpu_time_used);

    const char* ipath = "./results/MGARD_Lagrange_expected/v2_1000/MGARD_uniform_raw/MGARD_uniform_4e15.npy";

    const char* datapath = "../XGC_2/dataset/";
    const char* readin_f = "../XGC_2/dataset/d3d_coarse_v2_1000.bp/";
    vector <double> i_f;
    read_xgc_file(readin_f, i_f, nnodes, nphi, nsize);

    vector <double> vol;
    vector <double> vth;
    vector <double> vp;
    vector <double> mu_qoi;
    vector <double> vth2;
    vector <double> vol_f;
    vector <double> vth_f;
    vector <double> vp_f;
    vector <double> mu_qoi_f;
    vector <double> vth2_f;
    double sml_e_charge = 1.6022e-19;
    double ptl_mass = 3.344e-27;
    int iphi,i;
    // get the output of xgcexp.f0_diag_vol_origdim
    vector<unsigned long> vol_data_shape;
    load_npy_file("./vol.npy", vol_f, vol_data_shape);
    vector<unsigned long> vth_data_shape;
    load_npy_file("./vth.npy", vth_f, vth_data_shape);
    vector<unsigned long> vp_data_shape;
    load_npy_file("./vp.npy", vp_f, vp_data_shape);
    vector<unsigned long> mu_qoi_data_shape;
    load_npy_file("./mu_qoi.npy", mu_qoi_f, mu_qoi_data_shape);
    vector<unsigned long> vth2_data_shape;
    load_npy_file("./vth2.npy", vth2_f, vth2_data_shape);
    read_f0_params(datapath, i_f, nnodes, nphi, nsize, vol, vth, vp,
        mu_qoi, vth2, sml_e_charge, ptl_mass);
    for (i=0; i<vol_f.size(); ++i) {
        if (abs(vol_f[i]-vol[i]) > 10) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, vol_f[i], vol[i]);
        }
    }
    for (i=0; i<vth_f.size(); ++i) {
        if (abs(vth_f[i]-vth[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, vth_f[i], vth[i]);
        }
    }
    for (i=0; i<vp_f.size(); ++i) {
        if (abs(vp_f[i]-vp[i]) > 5) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, vp_f[i], vp[i]);
        }
    }
    for (i=0; i<mu_qoi_f.size(); ++i) {
        if (abs(mu_qoi_f[i]-mu_qoi[i]) > 0.01) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, mu_qoi_f[i], mu_qoi[i]);
        }
    }

    // get the actual QoIs
    vector <double> den_f;
    vector <double> upara_f;
    vector <double> tperp_f;
    vector <double> tpara_f;
    vector <double> den_f_data;
    vector <double> upara_f_data;
    vector <double> tperp_f_data;
    vector <double> tpara_f_data;
    vector<unsigned long> den_f_data_shape;
    load_npy_file("./den_f.npy", den_f_data, den_f_data_shape);
    vector<unsigned long> upara_f_data_shape;
    load_npy_file("./upara_f.npy", upara_f_data, upara_f_data_shape);
    vector<unsigned long> tperp_f_data_shape;
    load_npy_file("./tperp_f.npy", tperp_f_data, tperp_f_data_shape);
    vector<unsigned long> tpara_f_data_shape;
    load_npy_file("./tpara_f.npy", tpara_f_data, tpara_f_data_shape);
    for (iphi=0; iphi<nphi; ++iphi) {
        compute_C_qois(i_f, iphi, nnodes, nsize, vol, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge, den_f, upara_f, tperp_f, tpara_f, den_f_data, upara_f_data, tperp_f_data, tpara_f_data);
    }
#if 0
    for (i=0; i<den_f.size(); ++i) {
        if (abs(den_f_data[i]-den_f[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, den_f_data[i], den_f[i]);
        }
    }
    for (i=0; i<upara_f.size(); ++i) {
        if (abs(upara_f_data[i]-upara_f[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, upara_f_data[i], upara_f[i]);
        }
    }
    for (i=0; i<tperp_f.size(); ++i) {
        if (abs(tperp_f_data[i]-tperp_f[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, tperp_f_data[i], tperp_f[i]);
        }
    }
    for (i=0; i<tpara_f.size(); ++i) {
        if (abs(tpara_f_data[i]-tpara_f[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, tpara_f_data[i], tpara_f[i]);
        }
    }
#endif

    // Assign recon_breg, lagranges
    int count = 0;
    // double* recon_breg = (double*) malloc(n_phi*nnodes*nsize*nsize);
    // double* lagranges = (double*) malloc(n_phi*nnodes*4);
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
    vector <double> V2 = qoi_V2(vol, vth, vp, nnodes, nsize);
    vector <double> V3 = qoi_V3(vol, vth2, mu_qoi, ptl_mass, nnodes, nsize);
    vector <double> V4 = qoi_V4(vol, vth2, vp, ptl_mass, nnodes, nsize);
    vector <double> tpara_data;
    int p, idx;
    for (p=0; p<8; ++p) {
        for (i=0; i<nnodes; ++i) {
            tpara_data.push_back(sml_e_charge * tpara_f[nnodes*p + i] +
                vth2[i] * ptl_mass * pow((upara_f[nnodes*p + i]/vth[i]), 2));
        }
    }
    // compare vectors V2, V3, V4, and Tpara
    checkVectorsV(V2, V3, V4, tpara_data);
    start = clock();
    for (p=0; p<8; ++p) {
        double* D = &den_f[p*nnodes];
        double* U = &upara_f[p*nnodes];
        double* Tperp = &tperp_f[p*nnodes];
        double* Tpara = &tpara_data[p*nnodes];
        int count_unLag = 0;
        vector <int> node_unconv;
        double maxD = -99999;
        double maxU = -99999;
        double maxTperp = -99999;
        double maxTpara = -99999;
        for (i=0; i<nnodes; ++i) {
            if (D[i] > maxD) {
                maxD = D[i];
            }
            if (U[i] > maxU) {
                maxU = U[i];
            }
            if (Tperp[i] > maxTperp) {
                maxTperp = Tperp[i];
            }
            if (Tpara[i] > maxTpara) {
                maxTpara = Tpara[i];
            }
        }
        double DeB = pow(maxD*1e-09, 2);
        double UeB = pow(maxU*1e-09, 2);
        double TperpEB = pow(maxTperp*1e-09, 2);
        double TparaEB = pow(maxTpara*1e-09, 2);
        for (idx=0; idx<nnodes; ++idx) {
            double* recon_one = &recon[nnodes*nsize*nsize*p + nsize*nsize*idx];
            double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
            vector <double> L2_den;
            vector <double> L2_upara;
            vector <double> L2_tperp;
            vector <double> L2_tpara;
            count = 0;
            double aD = D[idx]*sml_e_charge;
            while (1) {
                for (i=0; i<nsize*nsize; ++i) {
                    K[i] = lambdas[0]*vol[nsize*nsize*idx + i] +
                           lambdas[1]*V2[nsize*nsize*idx + i] +
                           lambdas[2]*V3[nsize*nsize*idx + i] +
                           lambdas[3]*V4[nsize*nsize*idx + i];
                }
                double update_D=0, update_U=0, update_Tperp=0, update_Tpara=0;
                if (count > 0) {
                    for (i=0; i<nsize*nsize; ++i) {
                        breg_result[i] = recon_one[i]*
                            exp(-K[i]);
                        update_D += breg_result[i]*vol[
                            nsize*nsize*idx + i];
                        update_U += breg_result[i]*V2[
                            nsize*nsize*idx + i]/D[idx];
                        update_Tperp += breg_result[i]*V3[
                            nsize*nsize*idx + i]/aD;
                        update_Tpara += breg_result[i]*V4[
                            nsize*nsize*idx + i]/D[idx];
                    }
                    L2_den.push_back(pow((update_D - D[idx]), 2));
                    L2_upara.push_back(pow((update_U - U[idx]), 2));
                    L2_tperp.push_back(pow((update_Tperp-Tperp[idx]), 2));
                    L2_tpara.push_back(pow((update_Tpara-Tpara[idx]), 2));
                    // L2_den.push_back(update_D);
                    // L2_upara.push_back(update_U);
                    // L2_tperp.push_back(update_Tperp);
                    // L2_tpara.push_back(update_Tpara);
                    bool c1, c2, c3, c4;
                    bool converged = (isConverged(L2_den, DeB)
                        && isConverged(L2_upara, UeB)
                        && isConverged(L2_tpara, TparaEB)
                        && isConverged(L2_tperp, TperpEB));
                    if (converged) {
                        for (i=0; i<nsize*nsize; ++i) {
                            recon_breg.push_back(breg_result[i]);
                        }
                        lagranges.push_back(lambdas[0]);
                        lagranges.push_back(lambdas[1]);
                        lagranges.push_back(lambdas[2]);
                        lagranges.push_back(lambdas[3]);
                        if (idx % 2000 == 0) {
                            printf ("node %d finished\n", idx);
                        }
                        break;
                    }
                    else if (count == 20 && !converged) {
                        for (i=0; i<nsize*nsize; ++i) {
                            recon_breg.push_back(recon_one[i]);
                        }
                        lagranges.push_back(lambdas[0]);
                        lagranges.push_back(lambdas[1]);
                        lagranges.push_back(lambdas[2]);
                        lagranges.push_back(lambdas[3]);
                        printf ("Node %d did not converge\n", idx);
                        count_unLag = count_unLag + 1;
                        node_unconv.push_back(idx);
                        break;
                    }
                }
                double gvalue1 = D[idx], gvalue2 = U[idx]*D[idx];
                double gvalue3 = Tperp[idx]*aD, gvalue4 = Tpara[idx]*D[idx];
                double hvalue1 = 0, hvalue2 = 0, hvalue3 = 0, hvalue4 = 0;
                double hvalue5 = 0, hvalue6 = 0, hvalue7 = 0;
                double hvalue8 = 0, hvalue9 = 0, hvalue10 = 0;

#if 0
                for (i=0; i<2*nsize; ++i) {
                    gvalue3 += recon_one[i]*
                          V3[nsize*nsize*idx + i]*exp(-K[i])*-1.0;
                    printf ("i=%d, gvalue = %g, recon = %g, vol = %g, exp = %g, den = %g\n", i, gvalue3, recon_one[i], V3[nsize*nsize*idx + i], exp(-K[i]),  Tperp[idx]*aD);
                }
#endif
                for (i=0; i<nsize*nsize; ++i) {
                    gvalue1 += recon_one[i]*vol[nsize*nsize*idx + i]*
                      exp(-K[i])*-1.0;
                    gvalue2 += recon_one[i]*
                      V2[nsize*nsize*idx + i]*exp(-K[i])*-1.0;
                    gvalue3 += recon_one[i]*
                      V3[nsize*nsize*idx + i]*exp(-K[i])*-1.0;
                    gvalue4 += recon_one[i]*
                      V4[nsize*nsize*idx + i]*exp(-K[i])*-1.0;

                    hvalue1 += recon_one[i]*pow(
                        vol[nsize*nsize*idx + i], 2)*exp(-K[i]);
                    hvalue2 += recon_one[i]* vol[
                        nsize*nsize*idx + i]*V2[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue3 += recon_one[i]* vol[
                        nsize*nsize*idx + i]*V3[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue4 += recon_one[i]* vol[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue5 += recon_one[i]*pow(
                        V2[nsize*nsize*idx + i],2)*exp(-K[i]);
                    hvalue6 += recon_one[i]*V2[
                        nsize*nsize*idx + i]*V3[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue7 += recon_one[i]*V2[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue8 += recon_one[i]*pow(
                        V3[nsize*nsize*idx + i],2)*exp(-K[i]);
                    hvalue9 += recon_one[i]*V3[
                        nsize*nsize*idx + i]*V4[nsize*nsize*idx + i]*
                        exp(-K[i]);
                    hvalue10 += recon_one[i]*pow(
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
                hessians[1][0] = hvalue2;
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
                int order = 4;
                int k;
                double d = determinant(hessians, order);
                if (d == 0) {
                    printf ("Need to define pesudoinverse for matrix in node %d\n", idx);
                }
                else{
                    double** inverse = cofactor(hessians, order);
                    double matmul[4] = {0, 0, 0, 0};
                    for (i=0; i<4; ++i) {
                        matmul[i] = 0;
                        for (k=0; k<4; ++k) {
                            matmul[i] += inverse[i][k] * gradients[k];
                        }
                    }
                    lambdas[0] = lambdas[0] - matmul[0];
                    lambdas[1] = lambdas[1] - matmul[1];
                    lambdas[2] = lambdas[2] - matmul[2];
                    lambdas[3] = lambdas[3] - matmul[3];
                }
                count = count + 1;
            }
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start))/CLOCKS_PER_SEC;
    printf ("time taken %g seconds", cpu_time_used);
    //compute reconstruction RMSE
    double recon_rmse = rmse_error(i_f, recon);
    //compute breg RMSE
    double breg_rmse = rmse_error(i_f, recon_breg);
    printf ("recon PD rmse: %g, breg PD rmse: %g\n", recon_rmse, breg_rmse);
    FILE* fp = fopen("recon_breg.txt", "w");
    for (int ii=0; ii<recon_breg.size(); ++ii) {
        fprintf (fp, "%d, %g\n", ii, recon_breg[ii]);
    }
    fclose(fp);

    fp = fopen("lagranges.bin", "wb");
    for (int ii=0; ii<lagranges.size(); ++ii) {
        fprintf (fp, "%d, %g\n", i, lagranges[ii]);
    }
    fclose(fp);


#if 0
    // compute compression ratio using 1 plane
    double data_size = nnodes * nsize * nsize * 64;
    double lagrange_cost = nnodes * 4 * 64;
    double compression_ratio = data_size/((data_size/
          mgard_compression_ratio) + lagrange_cost);

    printf ("compression_ratio_fixed_cost_MGARD_breg: %g\n",
        compression_ratio);

    // store Lagranges with PQ
#endif
}
