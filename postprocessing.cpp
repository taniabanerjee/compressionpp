#include <string.h>
#include <time.h>
#include "adios2.h"
#include "postprocessing.hpp"

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

void read_f0_params(const char* datapath, double* i_f,
    int nnodes, int nphi, int nsize, vector <double> &vol,
    vector <double> &vth, vector <double> &vp, vector <double> &mu_qoi,
    vector <double> &vth2, double &sml_e_charge, double &ptl_mass,
    vector <double> &den_ref)
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
    vector<std::size_t> ev_shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {ev_shape[0], ev_shape[1]}));
    std::vector<double> f0_T_ev;
    reader_vol.Get<double>(var_i_f_in, f0_T_ev);

    var_i_f_in = read_vol_io.InquireVariable<double>("f0_den");
    vector<std::size_t> den_shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0}, {den_shape[0]}));
    reader_vol.Get<double>(var_i_f_in, den_ref);
    reader_vol.Close();

    set_vol(grid_vol, f0_nvp[0], f0_nmu[0], nnodes, vol);
    set_vp(f0_nvp[0], f0_dvp[0], vp);
    set_mu_qoi(f0_nmu[0], f0_dsmu[0], mu_qoi);
    set_vth2(f0_T_ev, nnodes, sml_e_charge, ptl_mass, vth2, vth);
}

char* getDataPath(const char* filepath)
{
    char* datapath = strdup(filepath);
    char* del = &datapath[strlen(datapath)-1];

    while (del > datapath && *del == '/')
        del--;

    while (del > datapath && *del != '/')
        del--;

    if (*del == '/')
        *del = '\0';

    return datapath;
}

vector<double> compute_lagrange_parameters(const char* filepath, double* recon, int local_elements, int local_nodes, double* i_f, int nphi, int nnodes, int vx, int vy, int offset,  double* breg_recon, double &pd_error_b, double &pd_error_a, double &density_error_b, double &density_error_a, double &upara_error_b, double &upara_error_a, double &tperp_error_b, double &tperp_error_a, double &tpara_error_b, double &tpara_error_a)
{
    // Remove non-negative values from input
    clock_t start = clock();
    int ii;
    for (ii=0; ii<local_elements; ++ii) {
        if (!(recon[ii] > 0)) {
            recon[ii] = 100;
        }
    }
    // Compute f0_params;
    vector <double> vol, vth, vp, mu_qoi, vth2;
    double sml_e_charge, ptl_mass;
    // Set datapath
    char* datapath = getDataPath(filepath);

    // read_f0_params (datapath, i_f, nnodes, nphi, vx, vol, vth, vp,
        // mu_qoi, vth2, sml_e_charge, ptl_mass);
    vector <double> lagranges;
    return lagranges;
}

