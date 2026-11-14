# Optimization flag evaluation: `basic_optmization_nn.cpp`

| | |
|---|---|
| **File** | `basic_optmization_nn.cpp` |
| **Date/time** | 2026-11-14 19:59:52 -03 |
| **Mode** | Serial (g++) |
| **Compiler** | g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0 |
| **CPU** | Intel(R) Core(TM) Ultra 9 185H (22 logical cores) |
| **OS** | Linux 6.8.0-149-generic |
| **Git revision** | v0.1-Baseline-Data-4-gcf4fc4e-dirty |
| **Working directory** | `/home/galazzi/Documents/Insper_7_Semestre/Super_Computação/APS2/awful-barenn-Gabriel-Galazzi` |

## Summary

| Level | Flags | Compile | Compile time (s) | Run exit | Wall time (s) | Reported time (s) | Accuracy | Speedup vs none | Same output as none |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| none | `(none)` | ok | 0.575 | 0 | 4.432 | 4.3887 | 95.5% | 1.00x | baseline |
| -O1 | `-O1` | ok | 0.696 | 0 | 0.782 | 0.7541 | 95.5% | 5.82x | yes |
| -O2 | `-O2` | ok | 0.733 | 0 | 0.536 | 0.5051 | 95.5% | 8.69x | yes |
| -O3 | `-O3` | ok | 0.944 | 0 | 0.252 | 0.2241 | 95.5% | 19.58x | yes |
| -Ofast | `-Ofast` | ok | 0.947 | 0 | 0.184 | 0.1538 | 95.5% | 28.54x | **no** |

- Wall time covers the whole run (data loading included). Reported time is what the program prints.
- Speedup is relative to the `none` level and based on program-reported time.
- "Same output as none" compares the full output, skipping lines that match `[Tt]empo|[Tt]ime`.

## Detailed results

### none

```bash
g++ -std=c++17 basic_optmization_nn.cpp -o basic_optmization_nn
```

Compiled in 0.575 s. Run exited with code 0 after 4.432 s.

Program output:

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
Tempo de treino 4.3887 s

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

### -O1

```bash
g++ -std=c++17 -O1 basic_optmization_nn.cpp -o basic_optmization_nn
```

Compiled in 0.696 s. Run exited with code 0 after 0.782 s.

Program output:

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
Tempo de treino 0.7541 s

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

### -O2

```bash
g++ -std=c++17 -O2 basic_optmization_nn.cpp -o basic_optmization_nn
```

Compiled in 0.733 s. Run exited with code 0 after 0.536 s.

Program output:

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
Tempo de treino 0.5051 s

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

### -O3

```bash
g++ -std=c++17 -O3 basic_optmization_nn.cpp -o basic_optmization_nn
```

Compiled in 0.944 s. Run exited with code 0 after 0.252 s.

Program output:

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
Tempo de treino 0.2241 s

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

### -Ofast

```bash
g++ -std=c++17 -Ofast basic_optmization_nn.cpp -o basic_optmization_nn
```

Compiled in 0.947 s. Run exited with code 0 after 0.184 s.

Program output:

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
Epoca 41/51 | perda media 0.0156
Epoca 46/51 | perda media 0.0145
Epoca 51/51 | perda media 0.0135
Tempo de treino 0.1538 s

AVALIACAO FINAL
Acuracia 95.5% | 1716 acertos em 1797 imagens
Perda media 0.0436

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
       7         9           11
       8         1            6
       3         2            5
```
