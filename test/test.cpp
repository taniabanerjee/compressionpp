#include <stdio.h>
#include <stdlib.h>
#include <adios2.h>

using namespace std;
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
}

void read_vol(adios2::IO read_vol_io, adios2::Engine reader_vol,
    double f0_nvp, double f0_nmu, int nnodes, vector <double> &vol)
{
    adios2::Variable<double> var_i_f_in;
    var_i_f_in = read_vol_io.InquireVariable<double>("f0_grid_vol_vonly");
    size_t num_nodes = var_i_f_in.Shape()[1];
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, num_nodes}));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);

    std::vector<double> vp_vol;
    vp_vol.push_back(0.5);
    for (int ii = 0; ii<f0_nvp*2+1; ++ii) {
        vp_vol.push_back(1.0);
    }
    vp_vol.push_back(0.5);

    std::vector<double> mu_vol;
    mu_vol.push_back(0.5);
    for (int ii = 0; ii<f0_nmu+1; ++ii) {
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
            vol.push_back(grid_vol[ii] * mu_vp_vol[jj]);
        }
    }
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
    size_t num_nodes = var_i_f_in.Shape()[1];
    var_i_f_in.SetSelection(adios2::Box<adios2::Dims>({0, 0}, {1, num_nodes}  ));
    std::vector<double> grid_vol;
    reader_vol.Get<double>(var_i_f_in, grid_vol);

    printf ("%g\n", grid_vol[0]);

    reader_vol.Close();

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
    // read_mgard_file();

    const char* datapath = "/gpfs/alpine/csc143/scratch/tania/XGC_2/dataset/";
    const char* readin_f = "/gpfs/alpine/csc143/scratch/tania//XGC_2/dataset/d3d_coarse_v2_1000.bp/";
    vector <double> i_f;
    read_xgc_file(readin_f, i_f, nnodes, nphi, nsize);

    vector <double> vol;
    vector <double> vth;
    vector <double> vp;
    vector <double> mu_qoi;
    vector <double> vth2;
    double sml_e_charge;
    double ptl_mass;
    read_f0_params(datapath, i_f, nnodes, nphi, nsize, vol, vth, vp,
        mu_qoi, vth2, sml_e_charge, ptl_mass);
}
