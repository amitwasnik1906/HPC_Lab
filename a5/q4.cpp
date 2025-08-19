// prefix_sum.cpp
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main() {
    int n = 20000000;
    vector<double> a(n, 1.0);  // or random values
    vector<double> s(n, 0.0);

    int T = 1;
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            T = omp_get_num_threads();
    }

    vector<double> block_sum(T, 0.0);

    double t0 = omp_get_wtime();

    // Phase 1: local scan and block sums
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = (long)tid * n / T;
        int end   = (long)(tid + 1) * n / T;
        double local = 0.0;
        for (int i = start; i < end; ++i) {
            local += a[i];
            s[i] = local;  // local inclusive scan
        }
        block_sum[tid] = local;
    }

    // Phase 2: prefix of block sums (sequential, T is small)
    double acc = 0.0;
    for (int t = 0; t < T; ++t) {
        double tmp = block_sum[t];
        block_sum[t] = acc;  // block_sum[t] holds offset for block t
        acc += tmp;
    }

    // Phase 3: add offsets to each block (except first)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int start = (long)tid * n / T;
        int end   = (long)(tid + 1) * n / T;
        double offset = block_sum[tid];
        if (offset != 0.0) {
            for (int i = start; i < end; ++i) {
                s[i] += offset;
            }
        }
    }

    double t1 = omp_get_wtime();
    cout << "Time: " << (t1 - t0) << " sec\n";
    cout << "s[0]=" << s[0] 
         << " s[n-1]=" << s[n-1] << "\n";

    return 0;
}
