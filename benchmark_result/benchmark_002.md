# Second Benchmark on my MacBook :-(

Well, that changes the picture. :-) This time I repeated the benchmark
with a recent development version of OpenBLAS (`develop`, commit `9d78ff857`,
`v0.3.34-641`), built from source. Unlike Julia's bundled OpenBLAS, this
version has SME2 kernels for DGEMM. And it really uses them on my M5:
OpenBLAS detects the `vortexm4` core, and breakpoints confirmed that
`dgemm_` reaches `sme_dgemm_kernel_VORTEXM4`, not a NEON kernel. The
[build record](../bench/results/2026-10-31-apple-m5-openblas-dev-build.txt)
contains the compiler, build command, and kernel checks.

I used the [same benchmark script](../bench/gemm_threads.jl) as before,
this time pointing it at the locally built OpenBLAS:

```sh
cd bench
OPENBLAS_LIBRARY=/path/to/libopenblas.dylib OPENBLAS_SUFFIX= \
    ULMBLAS_LIBRARY=/path/to/libulmblas_lbt.dylib julia --project=. gemm_threads.jl
```

(`OPENBLAS_SUFFIX=` because this build has no `64_` suffix on its symbol
names.)

Here is the output from a run on my M5 with the machine otherwise quiet:

```text
library of ulmBLAS: /Users/lehn/git/ulmBLAS.jl/dev/lib/libulmblas_lbt.dylib
LBT after `using ulmBLAS`: dgemm_ -> ulmBLAS
LBT for this benchmark:   dgemm_ -> OpenBLAS (libopenblas.dylib)
ulmBLAS.mul!:             ulmblas_dgemm_strided of libulmblas_lbt.dylib
dgemm_ calls through LBT: LinearAlgebra.mul! strided 0, contiguous 1; ulmBLAS.mul! 0
CPU: Apple M5 (4 performance + 6 efficiency cores), Julia 1.13.3, Julia threads 1, OpenBLAS libopenblas.dylib
OpenBLAS: OpenBLAS 0.3.34.dev  USE64BITINT DYNAMIC_ARCH NO_AFFINITY vortexm4 MAX_THREADS=64; core: vortexm4

case                            n     backend                   threads      ms    GFLOPS   bytes allocs
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     4.9     410.6       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   253.9       7.9      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        1     4.3     467.3       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.6     430.2       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    17.9     377.5       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   927.7       7.3      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        1    15.6     432.8       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    15.8     426.2       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.1     393.7       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   269.0       7.4      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        2     4.5     448.3       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.7     423.6       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    18.3     369.7       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   964.3       7.0      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        2    16.0     422.0       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.1     419.1       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.1     391.0       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   265.0       7.5      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        3     4.5     443.7       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.7     427.2       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    17.9     376.6       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   935.3       7.2      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        3    15.6     432.1       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.1     419.4       0      0
ulmBLAS.mul!, strided           1000  ulmBLAS (1 thread)              1     5.2     388.1       0      0
LinearAlgebra.mul!, strided     1000  generic Julia (1 thread)        1   263.1       7.6      64      1
LinearAlgebra.mul!, contiguous  1000  OpenBLAS                        4     4.5     449.0       0      0
ulmBLAS.mul!, contiguous        1000  ulmBLAS (1 thread)              1     4.7     427.8       0      0
ulmBLAS.mul!, strided           1500  ulmBLAS (1 thread)              1    18.1     373.4       0      0
LinearAlgebra.mul!, strided     1500  generic Julia (1 thread)        1   941.9       7.2      64      1
LinearAlgebra.mul!, contiguous  1500  OpenBLAS                        4    15.7     430.2       0      0
ulmBLAS.mul!, contiguous        1500  ulmBLAS (1 thread)              1    16.0     421.0       0      0
```

![GFLOPS of ulmBLAS.mul! and LinearAlgebra.mul! with OpenBLAS develop (SME2) for strided and contiguous matrices, n = 1000 and 1500, 1 to 4 OpenBLAS threads, on an Apple M5](../bench/results/2026-10-31-apple-m5-openblas-dev.svg)

## Some notes on that

- **Now we're talking.** With SME2 on both sides, OpenBLAS reaches
  430–467 GFLOPS for contiguous matrices, versus 419–430 GFLOPS for
  ulmBLAS. At n = 1000, OpenBLAS is about 9% faster with one thread;
  at n = 1500, they're within 3%. My big lead from the first test is gone. :-)
- **But where are the other threads?** Changing OpenBLAS's thread setting
  from 1 to 4 barely changes the results. The complete run used about
  one CPU core (346 seconds of CPU time over 345 seconds of wall time),
  and the SME2 kernel was entered once per call, even with four threads
  requested. So the `threads` column records the *requested setting*,
  not the number of cores actually doing work. This observation applies
  to this build and these measurements; it isn't a general claim about
  all OpenBLAS kernels.
- **Arbitrary strides still matter.** `LinearAlgebra.mul!` makes no
  BLAS `dgemm_` call for the strided view, regardless of which OpenBLAS
  is selected. `ulmBLAS.mul!` continues to use its strided GEMM directly.
