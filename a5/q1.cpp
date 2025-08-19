// mm_mult.cpp
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    int N = 1024; // adjust for testing

    vector<double> A(N * N, 1.0);
    vector<double> B(N * N, 1.0);
    vector<double> C(N * N, 0.0);

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }

    double t1 = omp_get_wtime();
    cout << "Time: " << (t1 - t0) << " sec\n";

    // checksum
    double sum = 0.0;
    for (int i = 0; i < N * N; i++) sum += C[i];
    cout << "Checksum: " << sum << "\n";

    return 0;
}
