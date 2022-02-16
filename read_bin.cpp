#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char** argv)
{
    streampos size;
    char * memblock;

    int i=0;
    int offset = 0;
    for (i=0; i<6; ++i) {
        int rank = i;
        offset = offset + size/8;
        char ifilename [1024];
        char ofilename [1024];
        sprintf (ifilename, "xgc.mgard.rank%d_0.bin", rank);
        sprintf (ofilename, "output.rank%d_0.txt", rank);
        ifstream file (ifilename, ios::in|ios::binary|ios::ate);
        ofstream txtfile (ofilename);
        if (file.is_open())
        {
            size = file.tellg();
        
            cout << "size=" << size << "\n"; 
        
            memblock = new char [size];
            file.seekg (0, ios::beg);
            file.read (memblock, size);
            file.close();
        
            cout << "the entire file content is in memory \n";
            double* double_values = (double*)memblock;//reinterpret as doubles
            int dsize = size/8;
            if (txtfile.is_open()) {
                for(int k=0; k<dsize; k++) {
                    double value = double_values[k];
                    txtfile << "value ("<<(k+offset)<<")=" << value << "\n";
                }
            }
        }
    }
}
