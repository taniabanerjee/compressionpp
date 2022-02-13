#include <string.h>
#include <time.h>
#include "adios2.h"
#include "postprocessing.hpp"
// #define UF_DEBUG 0

#ifdef UF_DEBUG
#include "npy.hpp"
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
#endif

double determinant(double a[4][4], double k);
double** cofactor(double num[4][4], double f);
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

void read_f0_params(const char* datapath, unsigned long nnodes,
    unsigned long offset, int nphi, int nsize, vector <double> &vol,
    vector <double> &vth, vector <double> &vp, vector <double> &mu_qoi,
    vector <double> &vth2, double &sml_e_charge, double &ptl_mass)
{
    sml_e_charge = 1.6022e-19;
    ptl_mass = 3.344e-27;

    adios2::ADIOS ad;
    adios2::IO read_vol_io = ad.DeclareIO("xgc_vol");
    char vol_file[2048];
    sprintf(vol_file, "%s/xgc.f0.mesh.bp", datapath);
    adios2::Engine reader_vol = read_vol_io.Open(vol_file, adios2::Mode::Read);

    adios2::Variable<double> var_i_f_in;
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    vector<std::size_t> vol_shape = var_i_f_in.Shape();
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, offset}, {vol_shape[0], nnodes}));
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
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, offset}, {ev_shape[0], nnodes}));
    std::vector<double> f0_T_ev;
    reader_vol.Get<double>(var_i_f_in, f0_T_ev);
    reader_vol.Close();

    set_vol(grid_vol, f0_nvp[0], f0_nmu[0], nnodes, vol);
    set_vp(f0_nvp[0], f0_dvp[0], vp);
    set_mu_qoi(f0_nmu[0], f0_dsmu[0], mu_qoi);
    set_vth2(f0_T_ev, nnodes, sml_e_charge, ptl_mass, vth2, vth);
#ifdef UF_DEBUG
    printf ("vol %d, vp %d, mu_qoi %d, vth2 %d\n", vol.size(), vp.size(), mu_qoi.size(), vth.size());
#endif
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

void compute_C_qois(double* i_f, int iphi, int nnodes, int nsize,
    vector <double> &vol, vector <double> &vth,
    vector <double> &vp, vector <double> &mu_qoi, vector <double> &vth2,
    double ptl_mass, double sml_e_charge, vector <double> &den_f,
    vector <double> &upara_f, vector <double> &tperp_f,
    vector <double> &tpara_f)
{
    vector <double> den;
    vector <double> upar;
    vector <double> upar_;
    vector <double> tper;
    vector <double> en;
    vector <double> T_par;
    int i, j, k;
    double* f0_f = &i_f[iphi*nnodes*nsize*nsize];
#ifdef UF_DEBUG
    printf("f0[0] %g, f0[1] %g, i_f[index] %g, i_f[index+1] %g\n", f0_f[0], f0_f[1], i_f[iphi*nnodes*nsize*nsize], i_f[iphi*nnodes*nsize*nsize+1]);
#endif
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
            for (k=0; k<nsize; ++k) {
                tper.push_back(f0_f[nsize*nsize*i + nsize*j + k] *
                    vol[nsize*nsize*i + nsize*j + k] * 0.5 *
                    mu_qoi[j] * vth2[i] * ptl_mass);
            }
    }
    for (i=0; i<nnodes; ++i) {
        double value = 0;
        for (j=0; j<nsize*nsize; ++j) {
            value += tper[nsize*nsize*i + j];
        }
        tperp_f.push_back(value/den_f[den_index + i]/sml_e_charge);
        // printf ("Tperp %g, %g, %g, %g\n", value/den_f[den_index + i]/sml_e_charge, value, ptl_mass, sml_e_charge);
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

vector <double> qoi_V2(vector <double> &vol, vector <double> &vth,
    vector <double> &vp, int nnodes, int nsize, int offset)
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

vector<double> compute_lagrange_parameters(const char* filepath, double* recon, int local_elements, int local_nnodes, double* i_f, int nphi, int nnodes, int vx, int vy, int offset, double maxv, double* &breg_recon, double &pd_error_b, double &pd_error_a, double &density_error_b, double &density_error_a, double &upara_error_b, double &upara_error_a, double &tperp_error_b, double &tperp_error_a, double &tpara_error_b, double &tpara_error_a)
{
    // Remove non-negative values from input
    clock_t start = clock();
    int ii, i, j, k;
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

    // Revisit for precise params
    read_f0_params (datapath, local_nnodes, offset, nphi, vx, vol, vth, vp,
        mu_qoi, vth2, sml_e_charge, ptl_mass);

#ifdef UF_DEBUG 
    // Verify vol numbers
    vector <double> vol_f;
    vector<unsigned long> vol_data_shape;
    load_npy_file("./vol.npy", vol_f, vol_data_shape);
    double* vol_ref = new double[local_nnodes*vx*vy];
    k = 0;
    for (i=offset*vx*vy; i<(offset+local_nnodes)*vx*vy; i++) {
        vol_ref[k++] = vol_f[i];
    }
    for (i=0; i<vol.size(); ++i) {
        if (abs(vol[i]-vol_ref[i]) > 10) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, vol[i], vol_ref[i]);
        }
    }
    // Verify vth numbers
    vector <double> vth_f;
    vector<unsigned long> vth_data_shape;
    load_npy_file("./vth.npy", vth_f, vth_data_shape);
    double* vth_ref = new double[local_nnodes];
    k = 0;
    for (i=offset; i<(offset+local_nnodes); i++) {
        vth_ref[k++] = vth_f[i];
    }
    for (i=0; i<vth.size(); ++i) {
        if (abs(vth[i]-vth_ref[i]) > 10) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, vth[i], vth_ref[i]);
        }
    }
#endif

    vector <double> den_f, upara_f, tperp_f, tpara_f;
    int iphi = 0;
    for (iphi=0; iphi<nphi; ++iphi) {
        compute_C_qois(i_f, iphi, local_nnodes, vx, vol, vth, vp, mu_qoi, vth2, ptl_mass, sml_e_charge, den_f, upara_f, tperp_f, tpara_f);
    }
#ifdef UF_DEBUG
    printf ("den_f %d, upara_f %d, tperp_f %d, tpara_f %d\n", den_f.size(), upara_f.size(), tperp_f.size(), tpara_f.size());
    // Verify density numbers
    vector <double> den_f_data;
    vector<unsigned long> den_f_data_shape;
    load_npy_file("./den_f.npy", den_f_data, den_f_data_shape);
    double* den_ref = new double[nphi*local_nnodes];
    k = 0;
    for (i=0; i<den_f_data.size(); i+=16395) {
        for (j=0; j<local_nnodes; ++j) {
            den_ref[k++] = den_f_data[i+j+offset];
        }
    }
    for (i=0; i<den_f.size(); ++i) {
        if (abs(den_f[i]-den_ref[i]) > 1e+13) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, den_ref[i], den_f[i]);
        }
    }
    // Verify upara numbers
    vector <double> upara_f_data;
    vector<unsigned long> upara_f_data_shape;
    load_npy_file("./upara_f.npy", upara_f_data, upara_f_data_shape);
    double* upara_ref = new double[nphi*local_nnodes];
    k = 0;
    for (i=0; i<upara_f_data.size(); i+=16395) {
        for (j=0; j<local_nnodes; ++j) {
            upara_ref[k++] = upara_f_data[i+j+offset];
        }
    }
    for (i=0; i<upara_f.size(); ++i) {
        if (abs(upara_f[i]-upara_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, upara_ref[i], upara_f[i]);
        }
    }
    // Verify Tperp numbers
    vector <double> tperp_f_data;
    vector<unsigned long> tperp_f_data_shape;
    load_npy_file("./tperp_f.npy", tperp_f_data, tperp_f_data_shape);
    double* tperp_ref = new double[nphi*local_nnodes];
    k = 0;
    for (i=0; i<tperp_f_data.size(); i+=16395) {
        for (j=0; j<local_nnodes; ++j) {
            tperp_ref[k++] = tperp_f_data[i+j+offset];
        }
    }
    for (i=0; i<tperp_f.size(); ++i) {
        if (abs(tperp_f[i]-tperp_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, tperp_ref[i], tperp_f[i]);
        }
    }
    // Verify Tpara numbers
    vector <double> tpara_f_data;
    vector<unsigned long> tpara_f_data_shape;
    load_npy_file("./tpara_f.npy", tpara_f_data, tpara_f_data_shape);
    double* tpara_ref = new double[nphi*local_nnodes];
    k = 0;
    for (i=0; i<tpara_f_data.size(); i+=16395) {
        for (j=0; j<local_nnodes; ++j) {
            tpara_ref[k++] = tpara_f_data[i+j+offset];
        }
    }
    for (i=0; i<tpara_f.size(); ++i) {
        if (abs(tpara_f[i]-tpara_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, tpara_ref[i], tpara_f[i]);
        }
    }
#endif

    vector <double> lagranges;
    int count = 0;
    double gradients[4] = {0.0, 0.0, 0.0, 0.0};
    double hessians[4][4] = {0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0};
    double K[vx*vy];
    double breg_result[vx*vy];
    memset(K, 0, vx*vy*sizeof(double));
    vector <double> V2 = qoi_V2(vol, vth, vp, local_nnodes, vx, offset);
    vector <double> V3 = qoi_V3(vol, vth2, mu_qoi, ptl_mass, local_nnodes, vx);
    vector <double> V4 = qoi_V4(vol, vth2, vp, ptl_mass, local_nnodes, vx);
    vector <double> tpara_data;
    int idx;
    for (iphi=0; iphi<nphi; ++iphi) {
        for (i=0; i<local_nnodes; ++i) {
            tpara_data.push_back(sml_e_charge * tpara_f[local_nnodes*iphi + i] +
                vth2[i] * ptl_mass * pow((upara_f[local_nnodes*iphi + i]/vth[i]), 2));
        }
    }
#ifdef UF_DEBUG
    // Verify V2 numbers
    vector <double> V2_data;
    vector<unsigned long> V2_data_shape;
    load_npy_file("./V2.npy", V2_data, V2_data_shape);
    double* V2_ref = new double[local_nnodes*vx*vy];
    k = 0;
    for (i=offset*vx*vy; i<(offset+local_nnodes)*vx*vy; i++) {
        V2_ref[k++] = V2_data[i];
    }
    for (i=0; i<V2.size(); ++i) {
        if (abs(V2[i]-V2_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V2_ref[i], V2[i]);
        }
    }
    // Verify V3 numbers
    vector <double> V3_data;
    vector<unsigned long> V3_data_shape;
    load_npy_file("./V3.npy", V3_data, V3_data_shape);
    double* V3_ref = new double[local_nnodes*vx*vy];
    k = 0;
    for (i=offset*vx*vy; i<(offset+local_nnodes)*vx*vy; i++) {
        V3_ref[k++] = V3_data[i];
    }
    for (i=0; i<V3.size(); ++i) {
        if (abs(V3[i]-V3_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V3_ref[i], V3[i]);
        }
    }
    // Verify V4 numbers
    vector <double> V4_data;
    vector<unsigned long> V4_data_shape;
    load_npy_file("./V4.npy", V4_data, V4_data_shape);
    double* V4_ref = new double[local_nnodes*vx*vy];
    k = 0;
    for (i=offset*vx*vy; i<(offset+local_nnodes)*vx*vy; i++) {
        V4_ref[k++] = V4_data[i];
    }
    for (i=0; i<V4.size(); ++i) {
        if (abs(V4[i]-V4_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, V4_ref[i], V4[i]);
        }
    }
    // Verify Tpara numbers
    vector <double> Tpara_data;
    vector<unsigned long> Tpara_data_shape;
    load_npy_file("./Tpara.npy", Tpara_data, Tpara_data_shape);
    double* Tpara_ref = new double[local_nnodes*vx*vy];
    k = 0;
    for (i=offset*vx*vy; i<(offset+local_nnodes)*vx*vy; i++) {
        Tpara_ref[k++] = Tpara_data[i];
    }
    for (i=0; i<tpara_data.size(); ++i) {
        if (abs(tpara_data[i]-Tpara_ref[i]) > 100) {
            printf("i = %d, data-read = %g, data-computed= %g\n", i, Tpara_ref[i], tpara_data[i]);
        }
    }
    printf ("V2 %g, V3 %g, V4 %g, tpara %g\n", V2[0], V3[0], V4[0], tpara_data[0]);
#endif
    int breg_index = 0;
    start = clock();
    for (iphi=0; iphi<nphi; ++iphi) {
        double* D = &den_f[iphi*local_nnodes];
        double* U = &upara_f[iphi*local_nnodes];
        double* Tperp = &tperp_f[iphi*local_nnodes];
        double* Tpara = &tpara_data[iphi*local_nnodes];
        int count_unLag = 0;
        vector <int> node_unconv;
        double maxD = -99999;
        double maxU = -99999;
        double maxTperp = -99999;
        double maxTpara = -99999;
        for (i=0; i<local_nnodes; ++i) {
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
        double PDeB = pow(maxv*1e-09, 2);
        double* f0_f = &i_f[iphi*local_nnodes*vx*vy];
        for (idx=0; idx<local_nnodes; ++idx) {
            double* recon_one = &recon[local_nnodes*vx*vy*iphi + vx*vy*idx];
            double lambdas[4] = {0.0, 0.0, 0.0, 0.0};
            vector <double> L2_den;
            vector <double> L2_upara;
            vector <double> L2_tperp;
            vector <double> L2_tpara;
            vector <double> L2_PD;
            count = 0;
            double aD = D[idx]*sml_e_charge;
            while (1) {
                for (i=0; i<vx*vy; ++i) {
                    K[i] = lambdas[0]*vol[vx*vy*idx + i] +
                           lambdas[1]*V2[vx*vy*idx + i] +
                           lambdas[2]*V3[vx*vy*idx + i] +
                           lambdas[3]*V4[vx*vy*idx + i];
                }
#ifdef UF_DEBUG
                printf("L1 %g, L2 %g L3 %g, L4 %g K[0] %g\n", lambdas[0], lambdas[1], lambdas[2], lambdas[3], exp(-K[0]));
#endif
                double update_D=0, update_U=0, update_Tperp=0, update_Tpara=0, rmse_pd=0;
                if (count > 0) {
                    for (i=0; i<vx*vy; ++i) {
                        breg_result[i] = recon_one[i]*
                            exp(-K[i]);
                        update_D += breg_result[i]*vol[
                            vx*vy*idx + i];
                        update_U += breg_result[i]*V2[
                            vx*vy*idx + i]/D[idx];
                        update_Tperp += breg_result[i]*V3[
                            vx*vy*idx + i]/aD;
                        update_Tpara += breg_result[i]*V4[
                            vx*vy*idx + i]/D[idx];
                        rmse_pd += pow((breg_result[i] - f0_f[vx*vy*idx]), 2);
                    }
                    L2_den.push_back(pow((update_D - D[idx]), 2));
                    L2_upara.push_back(pow((update_U - U[idx]), 2));
                    L2_tperp.push_back(pow((update_Tperp-Tperp[idx]), 2));
                    L2_tpara.push_back(pow((update_Tpara-Tpara[idx]), 2));
                    L2_PD.push_back(sqrt(rmse_pd));
#ifdef UF_DEBUG
                    printf ("L2_den %g, %g\n", update_D, D[idx]);
#endif
                    bool c1, c2, c3, c4;
                    bool converged = (isConverged(L2_den, DeB)
                        && isConverged(L2_upara, UeB)
                        && isConverged(L2_tpara, TparaEB)
                        && isConverged(L2_tperp, TperpEB))
                        && isConverged(L2_PD, PDeB);
                    if (converged) {
                        for (i=0; i<vx*vy; ++i) {
                            breg_recon[breg_index++] = breg_result[i];
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
                        for (i=0; i<vx*vy; ++i) {
                            breg_recon[breg_index++] = recon_one[i];
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

                for (i=0; i<vx*vy; ++i) {
                    gvalue1 += recon_one[i]*vol[vx*vy*idx + i]*
                      exp(-K[i])*-1.0;
                    gvalue2 += recon_one[i]*
                      V2[vx*vy*idx + i]*exp(-K[i])*-1.0;
                    gvalue3 += recon_one[i]*
                      V3[vx*vy*idx + i]*exp(-K[i])*-1.0;
                    gvalue4 += recon_one[i]*
                      V4[vx*vy*idx + i]*exp(-K[i])*-1.0;

                    hvalue1 += recon_one[i]*pow(
                        vol[vx*vy*idx + i], 2)*exp(-K[i]);
                    hvalue2 += recon_one[i]* vol[
                        vx*vy*idx + i]*V2[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue3 += recon_one[i]* vol[
                        vx*vy*idx + i]*V3[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue4 += recon_one[i]* vol[
                        vx*vy*idx + i]*V4[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue5 += recon_one[i]*pow(
                        V2[vx*vy*idx + i],2)*exp(-K[i]);
                    hvalue6 += recon_one[i]*V2[
                        vx*vy*idx + i]*V3[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue7 += recon_one[i]*V2[
                        vx*vy*idx + i]*V4[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue8 += recon_one[i]*pow(
                        V3[vx*vy*idx + i],2)*exp(-K[i]);
                    hvalue9 += recon_one[i]*V3[
                        vx*vy*idx + i]*V4[vx*vy*idx + i]*
                        exp(-K[i]);
                    hvalue10 += recon_one[i]*pow(
                        V4[vx*vy*idx + i],2)*exp(-K[i]);
                }
                gradients[0] = gvalue1;
                gradients[1] = gvalue2;
                gradients[2] = gvalue3;
                gradients[3] = gvalue4;
#ifdef UF_DEBUG
                printf("Gradients: %g, %g, %g, %g\n", gradients[0], gradients[1], gradients[2], gradients[3]);
#endif
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
#if UF_DEBUG
                    printf("Hessians: %g, %g, %g, %g\n", hessians[0][0], hessians[0][1], hessians[0][2], hessians[0][3]);
                    printf("Inverse: %g, %g, %g, %g\n", inverse[0][0], inverse[0][1], inverse[0][2], inverse[0][3]);
#endif
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
    return lagranges;
}
