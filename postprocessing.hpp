#ifndef POST_PROCESSING_HPP
#define POST_PROCESSING_HPP

#include <vector>

using namespace std;

vector<double> compute_lagrange_parameters(const char* filepath, double* recon, int local_elements, int local_nnodes, double* i_f, int nphi, int nnodes, int vx, int vy, int offset, double maxv, double* &breg_recon, double &pd_error_b, double &pd_error_a, double &density_error_b, double &density_error_a, double &upara_error_b, double &upara_error_a, double &tperp_error_b, double &tperp_error_a, double &tpara_error_b, double &tpara_error_a);

#endif //POST_PROCESSING_HPP
