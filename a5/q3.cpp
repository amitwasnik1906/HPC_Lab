// mv_mult.cpp
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    int m = 20000, n = 1000;

    vector<double> A((size_t)m * n, 0.5);
    vector<double> x(n, 1.0);
    vector<double> y(m, 0.0);

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        size_t base = (size_t)i * n;
        for (int j = 0; j < n; ++j) {
            sum += A[base + j] * x[j];
        }
        y[i] = sum;
    }

    double t1 = omp_get_wtime();

    cout << "Time: " << (t1 - t0) << " sec\n";
    cout << "y[0]=" << y[0] 
         << " y[m-1]=" << y[m - 1] << "\n";

    return 0;
}
