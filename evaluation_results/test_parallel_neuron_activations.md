# OpenMP thread-scaling evaluation: `parallel_neuron_activations.cpp`

| | |
|---|---|
| **File** | `parallel_neuron_activations.cpp` (sha256 `c62454bdc54d`) |
| **Reference** | `Deprecated versions/reusable_buffers_nn.cpp` (sha256 `a552652767b9`, tag `v1.1-Buffers-Reutilizaveis`) |
| **Date/time** | 2026-11-25 19:08:15 -03 |
| **Mode** | OpenMP (g++ -fopenmp), compared with serial (g++) |
| **Compiler** | g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **CPU** | Intel(R) Core(TM) Ultra 9 185H (22 logical cores) |
| **OS** | Linux 6.8.0-149-generic |
| **Git revision** | v1.1-Buffers-Reutilizaveis-1-gbe0f47b-dirty |
| **OpenMP environment** | `OMP_PROC_BIND=close OMP_PLACES=cores`, `OMP_NUM_THREADS` as listed |
| **Method** | One run per configuration, all at `-O3`. No core pinning beyond `OMP_PLACES`, background load not controlled. Metric: program-reported training time (`Tempo de treino`). |

## Summary

| Build | Flags | Threads | Reported time (s) | Change vs reference | Accuracy | Same output as reference |
|---|---|---:|---:|---:|---:|---|
| reference (serial) | `-O3` | 1 | 0.2078 | baseline | 95.5% | baseline |
| parallel, no OpenMP | `-O3` | 1 | 0.2506 | +20.6% (1.21x) | 95.5% | yes |
| parallel, OpenMP | `-O3 -fopenmp` | 1 | 0.4282 | +106.1% (2.06x) | 95.5% | yes |
| parallel, OpenMP | `-O3 -fopenmp` | 2 | 0.5883 | +183.1% (2.83x) | 95.5% | yes |
| parallel, OpenMP | `-O3 -fopenmp` | 4 | 0.7963 | +283.2% (3.83x) | 95.5% | yes |
| parallel, OpenMP | `-O3 -fopenmp` | 8 | 1.2592 | +506.0% (6.06x) | 95.5% | yes |
| parallel, OpenMP | `-O3 -fopenmp` | 16 | 2.9826 | +1335.3% (14.35x) | 95.5% | yes |

- Change is relative to the serial reference; a positive value means slower.
- "Same output as reference" compares the full output, skipping lines that match `[Tt]empo|[Tt]ime`.
- "parallel, no OpenMP" is the same source compiled without `-fopenmp`: the pragmas are ignored and it runs serially. It isolates the cost of the code restructuring (two-pass `Backward`, column-wise reads of the weights in the `grad_in` pass, recomputed delta) from the cost of threading.

## Correctness under repeated runs

Repeated runs with different thread counts (default OpenMP placement), to catch intermittent data races.

| Threads | Runs | Reported times (s) |
|---:|---:|---|
| 3 | 5 | 0.7430 0.7256 0.7439 0.7613 0.7439 |
| 4 | 5 | 0.8695 0.8393 0.8395 0.8860 0.8835 |
| 6 | 5 | 1.0888 1.0845 1.1081 1.0996 1.0841 |
| 22 | 5 | 5.8106 7.7647 4.3096 4.3558 4.7923 |

- Runs whose output differs from the reference: **0 of 20**.

## Analysis

- Output is bit-identical to the serial reference for every thread count: each floating-point sum keeps the original order of its terms, and no `reduction` is used.
- Training time grows with the thread count. Each sample costs about 1 µs of arithmetic at `-O3` (3823 samples × 51 epochs), while the parallel version adds 9 synchronization points per sample (3 in `Forward`, 1 `single` for the loss, 5 in `Backward`), about 1.75 million barriers per run. The barrier cost exceeds the work it divides, and grows with team size.
- Layers have only 32, 16 and 10 neurons, so with many threads each one gets 1 to 2 neurons, or none.

## Detailed results

### Commands

```bash
g++ -std=c++17 -O3 "Deprecated versions/reusable_buffers_nn.cpp" -o reusable_buffers_nn
g++ -std=c++17 -O3 parallel_neuron_activations.cpp -o parallel_neuron_activations_serial
g++ -std=c++17 -O3 -fopenmp parallel_neuron_activations.cpp -o parallel_neuron_activations
OMP_NUM_THREADS=<n> OMP_PROC_BIND=close OMP_PLACES=cores ./parallel_neuron_activations
```

### Reference program output

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.2078 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```

### Parallel program output (OpenMP, 4 threads)

```text
TREINO | 3823 imagens
Epoca  1/51 | perda media 0.4129
Epoca  6/51 | perda media 0.0442
Epoca 11/51 | perda media 0.0312
Epoca 16/51 | perda media 0.0261
Epoca 21/51 | perda media 0.0227
Epoca 26/51 | perda media 0.0204
Epoca 31/51 | perda media 0.0184
Epoca 36/51 | perda media 0.0169
Epoca 41/51 | perda media 0.0155
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.7963 s

AVALIACAO FINAL
Acuracia 95.5% | 1717 acertos em 1797 imagens
Perda media 0.0432

ALGUMAS INFERENCIAS
Esperado  Previsto
       0         0
       1         1
       2         8
       3         3
       5         9
       7         9

CONFUSOES MAIS FREQUENTES
Esperado  Previsto  Ocorrencias
       7         9            9
       8         1            6
       3         2            5
```
