all:
	  mpicxx -g inverse_matrix.cpp postprocessing.cpp TestXGCAbsoluteError.cpp -o gpup -I/ccs/home/tania/MGARD-1.0.0/install/include -Wl,-rpath,/autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64 /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64/libadios2_cxx11_mpi.so.2.7.1 /autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64/libadios2_cxx11.so.2.7.1 -Wl,-rpath-link,/autofs/nccs-svm1_sw/summit/spack-envs/base/opt/linux-rhel8-ppc64le/gcc-7.5.0/adios2-2.7.1-qtvhwjxzdlbro3zv7zytmxm72fc3a5d5/lib64 -L/ccs/home/tania/MGARD-1.0.0/install/lib64 -lmgard -L/ccs/home/tania/MGARD-1.0.0/external/nvcomp/build/lib -lnvcomp -std=c++11 -I/ccs/home/tania/libnpy/include
