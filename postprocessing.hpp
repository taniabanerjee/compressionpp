#ifndef POST_PROCESSING_HPP
#define POST_PROCESSING_HPP

#include <vector>

using namespace std;

vector<double> compute_lagrange_parameters(const char* filepath, double* recon, int local_elements, int local_nnodes, double* i_f, int nphi, int nnodes, int vx, int vy, int offset, double maxv, double* &breg_recon, double* &before_errors, double* &after_errors);

#endif //POST_PROCESSING_HPP
