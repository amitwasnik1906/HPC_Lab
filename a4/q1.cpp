/* fib_openmp_tasks.cpp
   Compile: g++ -fopenmp fib_openmp_tasks.cpp -o fib
   Run: ./fib <n> <num_threads>
*/
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

vector<long long> memo;
vector<omp_lock_t> locks;

long long fib_task(int n) {
    if (n <= 1) return n;

    // Quick check without lock (optimistic)
    long long v = memo[n];
    if (v != -1) return v;

    // Try to acquire lock for this index
    omp_set_lock(&locks[n]);
    // Double-check now that we have lock
    if (memo[n] != -1) {
        v = memo[n];
        omp_unset_lock(&locks[n]);
        return v;
    }

    long long x = 0, y = 0;
    // Create tasks for children (tasking overhead cutoff)
    if (n > 20) { // cutoff to reduce overhead for small n
        #pragma omp task shared(x) firstprivate(n)
        x = fib_task(n - 1);
        #pragma omp task shared(y) firstprivate(n)
        y = fib_task(n - 2);
        #pragma omp taskwait
    } else {
        x = fib_task(n - 1);
        y = fib_task(n - 2);
    }

    memo[n] = x + y;
    v = memo[n];
    omp_unset_lock(&locks[n]);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <n> <num_threads>\n";
        return 1;
    }
    int n = stoi(argv[1]);
    int num_threads = stoi(argv[2]);

    memo.assign(n + 1, -1);
    locks.resize(n + 1);

    for (int i = 0; i <= n; i++) {
        omp_init_lock(&locks[i]);
    }
    memo[0] = 0;
    if (n >= 1) memo[1] = 1;

    omp_set_num_threads(num_threads);
    double t0 = omp_get_wtime();

    long long result = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fib_task(n);
        }
    }

    double t1 = omp_get_wtime();

    cout << "Fibonacci(" << n << ") = " << result << "\n";
    cout << "Threads used: " << num_threads << "\n";
    cout << "Time (s): " << (t1 - t0) << "\n";

    for (int i = 0; i <= n; i++) {
        omp_destroy_lock(&locks[i]);
    }

    return 0;
}
