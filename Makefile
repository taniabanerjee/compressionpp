MGARD_base=/gpfs/alpine/csc143/proj-shared/tania/MGARD-1.0.0
all:
	  mpicxx -O2 inverse_matrix.cpp postprocessing.cpp TestXGCAbsoluteError.cpp -o gpup -I$(MGARD_base)/install/include -Wl,-rpath,/autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64 /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64/libadios2_cxx11_mpi.so.2.7.1 /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64/libadios2_cxx11.so.2.7.1 -Wl,-rpath-link,/autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64 -L$(MGARD_base)/install/lib64 -lmgard -L$(MGARD_base)/external/nvcomp/build/lib -lnvcomp -std=c++11
