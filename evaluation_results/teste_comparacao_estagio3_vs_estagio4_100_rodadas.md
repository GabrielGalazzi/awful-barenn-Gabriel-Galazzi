# stage3 vs stage4: 100 paired rounds

| Level | stage3 median (s) | stage3 sd | stage4 median (s) | stage4 sd | Change in median | Median paired diff (s) [95% CI] | stage4 faster in | Sign test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 4.6993 | 0.0613 | 4.4417 | 0.0486 | -5.48% | -0.2805 [-0.3088, -0.2043] | 100/100 | 1.6e-30 |
| -O1 | 0.7292 | 0.0129 | 0.7439 | 0.0132 | +2.01% | +0.0115 [+0.0067, +0.0252] | 18/100 | 1e-10 |
| -O2 | 0.5030 | 0.0072 | 0.4945 | 0.0070 | -1.68% | -0.0069 [-0.0146, -0.0010] | 68/100 | 0.00016 |
| -O3 | 0.2240 | 0.0031 | 0.2195 | 0.0029 | -1.99% | -0.0034 [-0.0076, -0.0007] | 74/100 | 8.5e-07 |
| -Ofast | 0.1380 | 0.0020 | 0.1602 | 0.0040 | +16.09% | +0.0219 [+0.0209, +0.0239] | 0/100 | 1.6e-30 |

Per-core medians (s), to check that the core swap removed any core bias:

| Level | stage3 on CPU 1 | stage3 on CPU 3 | stage4 on CPU 1 | stage4 on CPU 3 |
|---|---:|---:|---:|---:|
| none | 4.7368 | 4.6627 | 4.4762 | 4.4152 |
| -O1 | 0.7327 | 0.7225 | 0.7505 | 0.7385 |
| -O2 | 0.5081 | 0.4985 | 0.4986 | 0.4909 |
| -O3 | 0.2258 | 0.2213 | 0.2219 | 0.2170 |
| -Ofast | 0.1390 | 0.1371 | 0.1620 | 0.1587 |

Correctness: in all 100 rounds both versions compiled and ran at every level, -O1/-O2/-O3 matched their own unoptimized output, and stage3 and stage4 printed identical output at every level (-Ofast included).
