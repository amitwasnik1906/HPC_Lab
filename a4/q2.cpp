/* prod_cons_openmp.cpp
   Compile: g++ -fopenmp prod_cons_openmp.cpp -o prodcons
   Run: ./prodcons <buffer_size> <items_to_produce> <num_producers> <num_consumers>
*/
#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0]
             << " <bufsize> <items> <producers> <consumers>\n";
        return 1;
    }

    int bufsize      = stoi(argv[1]);
    int total_items  = stoi(argv[2]);
    int nprod        = stoi(argv[3]);
    int ncons        = stoi(argv[4]);

    vector<int> buffer(bufsize);
    int in = 0, out = 0;
    int count = 0;

    omp_lock_t lock_in, lock_out;
    omp_init_lock(&lock_in);
    omp_init_lock(&lock_out);

    static int global_produced = 0;
    static int global_consumed = 0;

    omp_set_num_threads(nprod + ncons);
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        if (tid < nprod) {
            // Producer threads
            while (true) {
                int myglobal;
                #pragma omp atomic read
                myglobal = global_produced;

                if (myglobal >= total_items)
                    break;

                #pragma omp atomic update
                global_produced++;

                int item = myglobal + 1;

                int placed = 0;
                while (!placed) {
                    if (count < bufsize) {
                        omp_set_lock(&lock_in);
                        if (count < bufsize) {
                            buffer[in] = item;
                            in = (in + 1) % bufsize;
                            #pragma omp atomic update
                            count++;
                            placed = 1;
                            cout << "[Producer " << tid
                                 << "] produced " << item
                                 << " (count=" << count << ")\n";
                        }
                        omp_unset_lock(&lock_in);
                    }
                }
            }
        } else {
            // Consumer threads
            while (true) {
                int mycon;
                #pragma omp atomic read
                mycon = global_consumed;

                if (mycon >= total_items)
                    break;

                #pragma omp atomic update
                global_consumed++;

                int got = 0;
                int item = -1;
                while (!got) {
                    if (count > 0) {
                        omp_set_lock(&lock_out);
                        if (count > 0) {
                            item = buffer[out];
                            out = (out + 1) % bufsize;
                            #pragma omp atomic update
                            count--;
                            got = 1;
                            cout << "    [Consumer " << tid
                                 << "] consumed " << item
                                 << " (count=" << count << ")\n";
                        }
                        omp_unset_lock(&lock_out);
                    }
                }
            }
        }
    }

    double t1 = omp_get_wtime();
    cout << "All done. Time: " << (t1 - t0) << " sec\n";

    omp_destroy_lock(&lock_in);
    omp_destroy_lock(&lock_out);

    return 0;
}
