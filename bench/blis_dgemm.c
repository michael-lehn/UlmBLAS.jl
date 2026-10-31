/*
 *  blis_dgemm.c N: one call of dgemm_ (64-bit integers, as in BLIS built
 *  with configure -b 64) with random N x N matrices; used by
 *  check_blis_kernel.sh.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
void dgemm_(const char*, const char*, const int64_t*, const int64_t*, const int64_t*, const double*, const double*, const int64_t*, const double*, const int64_t*, const double*, double*, const int64_t*);
int main(int argc, char **argv) {
    int64_t n = atoi(argv[1]); double one = 1, zero = 0;
    double *A = malloc(n*n*8), *B = malloc(n*n*8), *C = malloc(n*n*8);
    for (int64_t i = 0; i < n*n; ++i) { A[i] = rand()/(double)RAND_MAX; B[i] = rand()/(double)RAND_MAX; }
    dgemm_("N", "N", &n, &n, &n, &one, A, &n, B, &n, &zero, C, &n);
    printf("n=%lld C[0]=%g\n", (long long)n, C[0]);
}
