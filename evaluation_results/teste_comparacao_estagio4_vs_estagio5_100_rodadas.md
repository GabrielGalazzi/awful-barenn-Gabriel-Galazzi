# stage4 vs stage5: 100 paired rounds

| | |
|---|---|
| **stage4** | `basic_optmization_nn.cpp` (sha256 `d51e323ed42c`) |
| **stage5** | `reusable_buffers_nn.cpp` (sha256 `a552652767b9`) |
| **Date/time** | 2026-11-15 14:10:59 -03 |
| **CPU** | Intel(R) Core(TM) Ultra 9 185H |
| **Compiler** | g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0, `g++ -std=c++17 <level>` |
| **OS** | Linux 6.8.0-149-generic |
| **Git revision** | v1.0-Otimizacoe-Basicas-Aplicadas-1-g3943ca9 |
| **Cores** | P-cores CPU 1 and CPU 3 (5.1 GHz max, distinct physical cores); governor `powersave`, EPP `performance`, AC power on |
| **Background load** | Firefox pinned to E-cores 12-19 for the whole run |
| **Method** | Each round runs stage4 and stage5 simultaneously, one per core; the cores swap every round, so each version runs 50 times on each core. One unrecorded warm-up pair per level. Metric: program-reported training time (`Tempo de treino`). |

| Level | stage4 median (s) | stage4 sd | stage5 median (s) | stage5 sd | Change in median | Median paired diff (s) [95% CI] | stage5 faster in | Sign test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 4.4706 | 0.0479 | 2.9536 | 0.0308 | -33.93% | -1.5128 [-1.5348, -1.4881] | 100/100 | 1.6e-30 |
| -O1 | 0.7517 | 0.0155 | 0.6558 | 0.0115 | -12.76% | -0.0956 [-0.0993, -0.0875] | 99/100 | 1.6e-28 |
| -O2 | 0.4986 | 0.0064 | 0.4945 | 0.0061 | -0.81% | -0.0021 [-0.0101, +0.0035] | 50/100 | 1 |
| -O3 | 0.2207 | 0.0026 | 0.2077 | 0.0026 | -5.87% | -0.0131 [-0.0153, -0.0103] | 100/100 | 1.6e-30 |
| -Ofast | 0.1603 | 0.0048 | 0.1313 | 0.0016 | -18.15% | -0.0294 [-0.0303, -0.0280] | 100/100 | 1.6e-30 |

Per-core medians (s), to check that the core swap removed any core bias:

| Level | stage4 on CPU 1 | stage4 on CPU 3 | stage5 on CPU 1 | stage5 on CPU 3 |
|---|---:|---:|---:|---:|
| none | 4.4944 | 4.4287 | 2.9557 | 2.9518 |
| -O1 | 0.7575 | 0.7417 | 0.6588 | 0.6532 |
| -O2 | 0.5031 | 0.4937 | 0.5001 | 0.4902 |
| -O3 | 0.2219 | 0.2187 | 0.2094 | 0.2059 |
| -Ofast | 0.1625 | 0.1590 | 0.1320 | 0.1305 |

Correctness (output compared with time lines removed):

- All 1000 runs exited with code 0 and reported a time: **yes**.
- Every version/level printed the same output in all 100 rounds: **yes**.
- stage4 and stage5 printed identical output at every level: none yes, -O1 yes, -O2 yes, -O3 yes, -Ofast yes.
- stage4 vs its own unoptimized output: -O1 same, -O2 same, -O3 same, -Ofast differs.
- stage5 vs its own unoptimized output: -O1 same, -O2 same, -O3 same, -Ofast differs.
