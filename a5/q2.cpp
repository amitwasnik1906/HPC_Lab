// mm_scalar.cpp
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    int rows = 4000, cols = 4000;
    double scalar = 2.5;

    vector<double> A(rows * cols);
    vector<double> B(rows * cols);

    // initialize A
    for (int i = 0; i < rows * cols; i++) {
        A[i] = i % 100;
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < rows * cols; ++idx) {
        B[idx] = scalar * A[idx];
    }

    double t1 = omp_get_wtime();

    cout << "Time: " << (t1 - t0) << " sec\n";
    cout << "Sample B[0]=" << B[0]
         << " B[last]=" << B[rows * cols - 1] << "\n";

    return 0;
}
