// Versão otimizada do porte CUDA (cuda_nn.cu).
//
// Estágio 1: fusão de kernels. Em cuda_nn.cu cada amostra disparava 12
// kernels (3 no Forward, 1 na perda, 8 no Backward) e o treino era limitado
// pelo custo fixo de cada lançamento. Aqui uma amostra inteira (Forward das 3
// camadas, perda e Backward com atualização dos pesos) roda em um único
// bloco. As etapas que antes eram kernels separados são separadas por
// __syncthreads(), e os valores intermediários (ativações, deltas e
// gradientes) ficam na memória compartilhada do bloco.
//
// Estágio 2: kernel persistente. Um lançamento por época em vez de um por
// amostra: o bloco percorre todas as amostras sem voltar ao host, com os
// pesos copiados para a memória compartilhada no início da época e de volta
// para a memória global no final.
//
// Estágio 3: dimensões em tempo de compilação. A atualização dos pesos
// recebe as dimensões da camada como parâmetros de template, e o laço
// sobre os pesos de cada thread é desenrolado.
//
// Estágio 4: redução em árvore com shuffle. Cada produto escalar do Forward
// e do grad_in é dividido entre várias threads de um warp, e as parciais
// são somadas com __shfl_xor_sync.
//
// Estágio 5: imagens em blocos na memória compartilhada. O treino copia 128
// imagens e seus rótulos de uma vez para a memória compartilhada, em vez de
// ler uma imagem da memória global no início de cada amostra.
//
// Estágio 6: fusão de etapas do Backward. O grad_in e a atualização de cada
// camada são uma etapa só (a thread lê o peso antigo, usa no grad_in e
// escreve o novo); o delta é calculado pela thread que termina o grad_in, o
// erro da camada final pela thread que termina o Forward, e a perda é somada
// por um warp que não participa do grad_in. De 11 para 6 barreiras por
// amostra.
//
// Estágio 7: cópia assíncrona das imagens com buffer duplo. O próximo
// bloco de 64 imagens é copiado com cp.async (__pipeline_memcpy_async)
// para um segundo buffer na memória compartilhada enquanto o bloco atual é
// processado, em vez de o bloco inteiro parar para copiar.
//
// Até o estágio 3 a ordem de todas as somas era a da CPU e a saída era bit a
// bit igual à de reusable_buffers_nn.cpp com --fmad=false. A redução em
// árvore soma as mesmas parcelas em outra ordem; em ponto flutuante isso
// muda o arredondamento, então os últimos dígitos da perda e algumas
// previsões de teste podem diferir da CPU, como acontece com a FMA. Os
// estágios seguintes mantêm as operações e a ordem das somas do estágio 5:
// a saída é bit a bit igual à dele.
//
// Compilação (RTX 4070 Laptop = sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false optimized_cuda_nn.cu -o optimized_cuda_nn
// --fmad=false impede que a*b+c vire uma FMA (um único arredondamento). Desde
// o estágio 4 a saída já não é bit a bit igual à da CPU; a flag continua
// para que a única diferença em relação à CPU seja a ordem das somas.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

using namespace std;

// Toda chamada da API CUDA passa por aqui. Erros viram exceção e são
// impressos pelo catch de main, como os demais erros do programa.
#define CUDA_CHECK(chamada)                                                    \
    do {                                                                       \
        cudaError_t erro_ = (chamada);                                         \
        if (erro_ != cudaSuccess) {                                            \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro_) + \
                                " em " + __FILE__ + ":" +                     \
                                to_string(__LINE__));                          \
        }                                                                      \
    } while (0)

// Lançamentos de kernel não retornam erro; ele é lido com cudaGetLastError.
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

namespace Barenn {

// Arquitetura fixa da rede: 64 -> 32 -> 16 -> 10.
constexpr int entradas = 64;
constexpr int n1 = 32;
constexpr int n2 = 16;
constexpr int classes = 10;

// Threads de cada bloco, no treino e na avaliação. A etapa mais larga é a
// atualização dos 2048 pesos da camada 1: 8 pesos por thread. Os produtos
// escalares da camada 1 usam 32 neurônios x 8 threads = 256 threads.
constexpr int threads_treino = 256;
// Imagens copiadas de uma vez da memória global para a memória
// compartilhada no treino. São dois buffers (buffer duplo): enquanto um é
// usado, o próximo bloco é copiado para o outro (2 x 64 x 64 floats = 32 KB).
constexpr int amostras_por_bloco = 64;

// Buffer na memória da GPU. O destrutor libera a memória, inclusive quando
// uma exceção interrompe o programa, então todo cudaMalloc tem seu cudaFree.
template <typename T>
class DeviceBuffer {
    T* ptr = nullptr;
    size_t n = 0;

public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : n(count) {
        if (n) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), n(o.n) {
        o.ptr = nullptr;
        o.n = 0;
    }

    T* get() { return ptr; }
    const T* get() const { return ptr; }
    size_t size() const { return n; }

    void upload(const T* host, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() { CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T))); }
};

// Ponteiros de device para os parâmetros de cada camada. A linha i (pesos
// do neurônio i) começa em w + i * ld. Na memória global ld = entradas da
// camada, o layout da CPU; na memória compartilhada as linhas ganham uma
// posição de folga, ver train_epoch_kernel.
struct Camada {
    float* w;
    float* b;
    int ld;
};

struct Parametros {
    Camada c1, c2, c3;
};

// ------------------------------------------------- etapas de uma camada --
// Cada função é executada por todas as threads_treino threads do bloco, que
// dividem o trabalho entre si. Nenhuma sincroniza o bloco: quem chama coloca
// __syncthreads() entre etapas que dependem uma da outra.

// Os produtos escalares são divididos entre S threads vizinhas de um mesmo
// warp: cada uma soma um trecho contíguo dos termos e as S parciais são
// combinadas em árvore com __shfl_xor_sync, que troca registradores entre
// threads do warp sem passar pela memória compartilhada. Na camada 1 a
// cadeia de somas dependentes de cada neurônio cai de 64 parcelas para 8
// parcelas mais 3 níveis de shuffle. Em troca, a ordem das somas deixa de
// ser a da CPU (ver o cabeçalho).
//
// Cada grupo de S threads precisa estar dentro de um warp e todas as threads
// de um warp que entra no shuffle precisam participar dele: S divide 32 e o
// número de threads usadas (OUT * S ou IN * S) é múltiplo de 32.

// z = viés + produto escalar da linha i com a entrada. As threads
// i * S .. i * S + S - 1 calculam o neurônio i; a primeira delas escreve o
// resultado: ReLU(z) nas camadas ocultas e, na camada final, z menos o alvo
// one-hot do rótulo, que é o erro usado na perda e o delta do Backward. Na
// avaliação rotulo = -1 e a saída é o próprio z (z - 0.0f == z).
template <int IN, int OUT, int S>
__device__ void forward_camada(const float* in, const Camada& c,
                               float* out, bool final, int rotulo = -1) {
    static_assert(32 % S == 0 && IN % S == 0 && (OUT * S) % 32 == 0 &&
                  OUT * S <= threads_treino, "divisão inválida");
    const int i = threadIdx.x / S;
    const int parte = threadIdx.x % S;
    if (i < OUT) {
        constexpr int trecho = IN / S;
        const float* row = c.w + i * c.ld + parte * trecho;
        const float* x = in + parte * trecho;
        float sum = 0.0f;
#pragma unroll
        for (int m = 0; m < trecho; ++m) sum += x[m] * row[m];
#pragma unroll
        for (int salto = S / 2; salto > 0; salto /= 2) {
            sum += __shfl_xor_sync(0xffffffffu, sum, salto);
        }
        if (parte == 0) {
            sum = c.b[i] + sum;
            // Mesma semântica de max(0.0f, sum) na CPU.
            out[i] = final ? sum - (i == rotulo ? 1.0f : 0.0f)
                           : (0.0f < sum ? sum : 0.0f);
        }
    }
}

// Backward de uma camada oculta em uma etapa só. As threads
// j * S .. j * S + S - 1 cuidam da coluna j de W, cada uma sobre OUT / S
// neurônios (arredondado para cima). Para cada peso W[i][j], a thread lê o
// valor antigo, soma delta[i] * W[i][j] ao grad_in[j] e escreve o peso
// atualizado; nenhuma outra thread toca esse peso na etapa, então não há
// corrida. A primeira thread de cada coluna termina o grad_in e já calcula o
// delta da camada anterior: grad_in[j] * ReLU'(z[j]), com z[j] > 0 sse
// last_input[j] = ReLU(z[j]) > 0.
template <int IN, int OUT, int S>
__device__ void backward_camada(const float* delta, const Camada& c,
                                const float* last_input, float learning_rate,
                                float* delta_in) {
    static_assert(32 % S == 0 && (IN * S) % 32 == 0 && IN * S <= threads_treino,
                  "divisão inválida");
    const int j = threadIdx.x / S;
    const int parte = threadIdx.x % S;
    if (j < IN) {
        constexpr int trecho = (OUT + S - 1) / S;
        const float entrada = last_input[j];
        float acc = 0.0f;
#pragma unroll
        for (int m = 0; m < trecho; ++m) {
            const int i = parte * trecho + m;
            if (OUT % S == 0 || i < OUT) {
                float& w = c.w[i * c.ld + j];
                const float d = delta[i];
                acc += d * w;
                w -= learning_rate * (d * entrada);
                if (j == 0) c.b[i] -= learning_rate * d;
            }
        }
#pragma unroll
        for (int salto = S / 2; salto > 0; salto /= 2) {
            acc += __shfl_xor_sync(0xffffffffu, acc, salto);
        }
        if (parte == 0) delta_in[j] = acc * (entrada > 0.0f ? 1.0f : 0.0f);
    }
}

// Uma iteração por peso. A thread que atualiza o peso j = 0 de cada linha
// também atualiza o viés. As dimensões da camada e o número de threads são
// constantes de compilação: o laço tem número fixo de iterações (8 na
// camada 1) e é desenrolado, então as leituras de todas as iterações são
// emitidas juntas em vez de cada iteração esperar a latência da anterior.
template <int IN, int OUT>
__device__ void update_camada(const Camada& c, const float* delta,
                              const float* last_input, float learning_rate) {
    constexpr int total = IN * OUT;
#pragma unroll
    for (int base = 0; base < total; base += threads_treino) {
        int idx = base + threadIdx.x;
        if (total % threads_treino == 0 || idx < total) {
            int i = idx / IN;
            int j = idx % IN;
            c.w[i * c.ld + j] -= learning_rate * (delta[i] * last_input[j]);
            if (j == 0) c.b[i] -= learning_rate * delta[i];
        }
    }
}

// Copia os pesos e vieses de uma camada, convertendo o layout das linhas
// (ld de origem para ld de destino).
__device__ void copiar_camada(const Camada& de, const Camada& para,
                              int in_features, int out_features) {
    for (int idx = threadIdx.x; idx < in_features * out_features; idx += blockDim.x) {
        int i = idx / in_features;
        int j = idx % in_features;
        para.w[i * para.ld + j] = de.w[i * de.ld + j];
    }
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) para.b[i] = de.b[i];
}

// ---------------------------------------------------------------- kernels --

// Linhas dos pesos na memória compartilhada. A memória compartilhada tem 32
// bancos de 4 bytes; no Forward a thread i lê w[i * ld + j]. Com ld = 64 (ou
// 32, ou 16) todas as threads do warp caem no mesmo banco e a leitura é
// serializada em até 32 acessos. Com uma posição de folga por linha
// (ld ímpar), threads vizinhas caem em bancos vizinhos. Compilar com
// -DFOLGA=0 remove a folga, para medir o efeito dos conflitos de banco.
#ifndef FOLGA
#define FOLGA 1
#endif
constexpr int ld1 = entradas + FOLGA;
constexpr int ld2 = n1 + FOLGA;
constexpr int ld3 = n2 + FOLGA;

// Thread que soma a perda de cada amostra. Fica no warp 1, que não participa
// do Backward da camada 3 (32 threads, warp 0), então a soma corre em
// paralelo com ele.
constexpr int thread_perda = 32;
static_assert(n2 * 2 <= thread_perda && thread_perda < threads_treino,
              "a thread da perda deve ficar fora do warp do Backward da camada 3");

// Uma época inteira de SGD em um bloco (kernel persistente). Os pesos são
// copiados para a memória compartilhada no início, as amostras são
// processadas em sequência, sem voltar ao host, e os pesos voltam para a
// memória global no final. Cada amostra faz as mesmas operações, com os
// mesmos operandos, que os 12 kernels por amostra de cuda_nn.cu.
__global__ void train_epoch_kernel(Parametros global,
                                   const float* __restrict__ imagens,
                                   const int* __restrict__ rotulos,
                                   int n, float learning_rate,
                                   double* __restrict__ epoch_loss) {
    __shared__ float w1[n1 * ld1], b1[n1];
    __shared__ float w2[n2 * ld2], b2[n2];
    __shared__ float w3[classes * ld3], b3[classes];
    // Dois blocos de imagens e rótulos (32.5 KB). As cópias de 16 bytes do
    // cp.async exigem destino alinhado a 16 bytes.
    __shared__ __align__(16) float bloco_imagens[2][amostras_por_bloco * entradas];
    __shared__ int bloco_rotulos[2][amostras_por_bloco];
    __shared__ float h1[n1], h2[n2], d3[classes], d2[n2], d1[n1];

    // Copia o bloco de imagens que começa na amostra `inicio` para o buffer
    // `buf` com cp.async: a cópia vai da memória global direto para a
    // compartilhada, sem passar pelos registradores, e a thread segue em
    // frente sem esperar. Cada thread copia 16 bytes (4 floats) por vez; o
    // início do bloco na memória global também é múltiplo de 16 bytes
    // (inicio * 64 floats). O commit fecha o grupo de cópias da thread,
    // que depois é esperado com __pipeline_wait_prior. Todas as threads
    // fazem o commit, mesmo sem cópia, para que a espera seja a mesma.
    auto copiar_bloco = [&](int inicio, int buf) {
        if (inicio < n) {
            const int quantas = min(amostras_por_bloco, n - inicio);
            const float* origem = imagens + size_t(inicio) * entradas;
            for (int q = 4 * threadIdx.x; q < quantas * entradas; q += 4 * threads_treino) {
                __pipeline_memcpy_async(&bloco_imagens[buf][q], origem + q, 16);
            }
            for (int q = threadIdx.x; q < quantas; q += threads_treino) {
                __pipeline_memcpy_async(&bloco_rotulos[buf][q], rotulos + inicio + q, 4);
            }
        }
        __pipeline_commit();
    };

    // A cópia do primeiro bloco corre em paralelo com a cópia dos pesos.
    copiar_bloco(0, 0);

    const Parametros p{{w1, b1, ld1}, {w2, b2, ld2}, {w3, b3, ld3}};
    copiar_camada(global.c1, p.c1, entradas, n1);
    copiar_camada(global.c2, p.c2, n1, n2);
    copiar_camada(global.c3, p.c3, n2, classes);

    // Perda acumulada da época; só a thread_perda a usa. Soma as mesmas
    // parcelas, na mesma ordem, que o total em memória global de cuda_nn.cu.
    double perda = 0.0;

    for (int k = 0; k < n; ++k) {
        const int pos = k % amostras_por_bloco;
        const int buf = (k / amostras_por_bloco) % 2;

        // No início de cada bloco, cada thread espera as suas cópias desse
        // bloco, pedidas amostras_por_bloco amostras antes; em geral já
        // terminaram e a espera é imediata. O __syncthreads() logo abaixo
        // torna as cópias de todas as threads visíveis ao bloco.
        if (pos == 0) __pipeline_wait_prior(0);

        // Este __syncthreads() fecha a amostra anterior.
        __syncthreads();
        const float* x = bloco_imagens[buf] + pos * entradas;

        forward_camada<entradas, n1, 8>(x, p.c1, h1, false);
        __syncthreads();

        // Pede a cópia do próximo bloco para o outro buffer, que fica em
        // segundo plano enquanto as amostras deste bloco são processadas. A
        // última leitura do outro buffer (a atualização da camada 1 da
        // amostra anterior) terminou antes do __syncthreads() do início da
        // amostra. O pedido vem depois do Forward da camada 1 e não antes:
        // medido, assim o treino fica 2% mais rápido, e o Forward da camada 1
        // começa sem esperar o cálculo dos endereços da cópia.
        if (pos == 0) copiar_bloco(k + amostras_por_bloco, 1 - buf);
        forward_camada<n1, n2, 16>(h1, p.c2, h2, false);
        __syncthreads();
        // Na camada final a saída já é o erro (escore - alvo), que é o delta
        // da camada 3.
        forward_camada<n2, classes, 16>(h2, p.c3, d3, true, bloco_rotulos[buf][pos]);
        __syncthreads();

        // Camada 3: grad_in com os pesos antigos, atualização e delta da
        // camada 2. Ao mesmo tempo, a thread_perda soma metade do erro
        // quadrático percorrendo as 10 classes na mesma ordem da CPU; nada
        // no Backward depende desse valor.
        backward_camada<n2, classes, 2>(d3, p.c3, h2, learning_rate, d2);
        if (threadIdx.x == thread_perda) {
            float total = 0.0f;
            for (int i = 0; i < classes; ++i) total += 0.5f * d3[i] * d3[i];
            perda += total;
        }
        __syncthreads();

        // Camada 2.
        backward_camada<n1, n2, 8>(d2, p.c2, h1, learning_rate, d1);
        __syncthreads();

        // Camada 1: grad_in não é usado, então só os pesos são atualizados.
        update_camada<entradas, n1>(p.c1, d1, x, learning_rate);
    }
    __syncthreads();

    copiar_camada(p.c1, global.c1, entradas, n1);
    copiar_camada(p.c2, global.c2, n1, n2);
    copiar_camada(p.c3, global.c3, n2, classes);
    if (threadIdx.x == thread_perda) *epoch_loss = perda;
}

// Forward de todo o conjunto de teste em um lançamento: as imagens são
// independentes, então cada bloco calcula os escores de uma imagem.
__global__ void inference_kernel(Parametros p,
                                 const float* __restrict__ imagens,
                                 float* __restrict__ escores) {
    __shared__ float x[entradas], h1[n1], h2[n2];
    const float* imagem = imagens + size_t(blockIdx.x) * entradas;

    for (int j = threadIdx.x; j < entradas; j += blockDim.x) x[j] = imagem[j];
    __syncthreads();
    forward_camada<entradas, n1, 8>(x, p.c1, h1, false);
    __syncthreads();
    forward_camada<n1, n2, 16>(h1, p.c2, h2, false);
    __syncthreads();
    forward_camada<n2, classes, 16>(h2, p.c3,
                                    escores + size_t(blockIdx.x) * classes, true);
}

// ------------------------------------------------------------------- rede --

// Todos os pesos e vieses em um único bloco de memória da GPU.
class Rede {
    static constexpr size_t tamanho_w1 = size_t(n1) * entradas;
    static constexpr size_t tamanho_w2 = size_t(n2) * n1;
    static constexpr size_t tamanho_w3 = size_t(classes) * n2;
    DeviceBuffer<float> memoria;

    // Pesos gerados no host com a mesma semente e a mesma distribuição da
    // CPU, que reinicia o gerador a cada camada.
    static void inicializar(float* destino, size_t n) {
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (size_t k = 0; k < n; ++k) destino[k] = distribution(gen);
    }

public:
    Parametros p;

    Rede() : memoria(tamanho_w1 + n1 + tamanho_w2 + n2 + tamanho_w3 + classes) {
        float* base = memoria.get();
        p.c1 = {base, base + tamanho_w1, entradas};
        p.c2 = {p.c1.b + n1, p.c1.b + n1 + tamanho_w2, n1};
        p.c3 = {p.c2.b + n2, p.c2.b + n2 + tamanho_w3, n2};

        // Vieses começam em zero; os pesos são sobrescritos logo abaixo.
        vector<float> h(memoria.size(), 0.0f);
        inicializar(h.data() + (p.c1.w - base), tamanho_w1);
        inicializar(h.data() + (p.c2.w - base), tamanho_w2);
        inicializar(h.data() + (p.c3.w - base), tamanho_w3);
        memoria.upload(h.data(), h.size());
    }
};

// ------------------------------------------------------------------ dados --

struct Amostra {
    array<float, entradas> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        int lidos = 0;
        Amostra amostra{};
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            amostra.imagem[lidos++] = valor / 16.0f;
        }
        if (lidos != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Conjunto residente na GPU: imagens em um bloco [n x 64] e rótulos em [n].
struct DeviceDataset {
    size_t n;
    DeviceBuffer<float> imagens;
    DeviceBuffer<int> rotulos;

    explicit DeviceDataset(const vector<Amostra>& dados)
        : n(dados.size()), imagens(dados.size() * entradas), rotulos(dados.size()) {
        vector<float> h_imagens(n * entradas);
        vector<int> h_rotulos(n);
        for (size_t k = 0; k < n; ++k) {
            copy(dados[k].imagem.begin(), dados[k].imagem.end(),
                 h_imagens.begin() + k * entradas);
            h_rotulos[k] = dados[k].rotulo;
        }
        imagens.upload(h_imagens.data(), h_imagens.size());
        rotulos.upload(h_rotulos.data(), h_rotulos.size());
    }

    const float* imagem(size_t k) const { return imagens.get() + k * entradas; }
    const int* rotulo(size_t k) const { return rotulos.get() + k; }
};

// Mesma perda da CPU, usada no host para o conjunto de teste.
float perda(const float* escores, int rotulo) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
    }
    return total;
}

void avaliar(const Rede& rede, const vector<Amostra>& teste, const DeviceDataset& d_teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Escores de todo o teste em [n x 10], calculados em um lançamento e
    // copiados para o host de uma vez.
    DeviceBuffer<float> d_escores(teste.size() * classes);
    inference_kernel<<<static_cast<unsigned>(d_teste.n), threads_treino>>>(
        rede.p, d_teste.imagens.get(), d_escores.get());
    CUDA_CHECK_KERNEL();
    vector<float> todos_escores(d_escores.size());
    d_escores.download(todos_escores.data(), todos_escores.size());

    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const float* escores = &todos_escores[i * classes];
        int previsto = static_cast<int>(max_element(escores, escores + classes) - escores);
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";
    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";
    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";
        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                 << "  " << setw(8) << erro.previsto
                 << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    // O terceiro argumento (épocas) existe só para encurtar perfis no Nsight;
    // o padrão continua 51.
    if (argc != 1 && argc != 3 && argc != 4) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste [epocas]]\n";
        return 1;
    }
    const string arquivo_treino = argc >= 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc >= 3 ? argv[2] : "optdigits.tes";
    const int epocas = argc == 4 ? stoi(argv[3]) : 51;

    try {
        if (epocas < 1) throw runtime_error("O número de épocas deve ser positivo.");
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        // Cria o contexto CUDA antes de qualquer medição; a primeira chamada
        // à API custa dezenas a centenas de milissegundos.
        CUDA_CHECK(cudaFree(nullptr));

        // Upload medido à parte, para a comparação com a CPU ser honesta.
        const auto inicio_upload = chrono::steady_clock::now();
        Barenn::DeviceDataset d_treino(treino);
        Barenn::DeviceDataset d_teste(teste);
        const chrono::duration<double> tempo_upload =
            chrono::steady_clock::now() - inicio_upload;

        constexpr float taxa = 0.02f;
        Barenn::Rede rede;
        Barenn::DeviceBuffer<double> d_perdas(epocas);
        d_perdas.zero();

        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double* perda_epoca = d_perdas.get() + epoca;
            Barenn::train_epoch_kernel<<<1, Barenn::threads_treino>>>(
                rede.p, d_treino.imagens.get(), d_treino.rotulos.get(),
                static_cast<int>(d_treino.n), taxa, perda_epoca);
            CUDA_CHECK_KERNEL();
        }
        // Os lançamentos são assíncronos: sem sincronizar, o cronômetro
        // pararia antes de a GPU terminar o trabalho.
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        vector<double> perdas(epocas);
        d_perdas.download(perdas.data(), perdas.size());
        for (double& p : perdas) p /= treino.size();

        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        cout << "Tempo de upload (H2D) " << tempo_upload.count() << " s\n";
        Barenn::avaliar(rede, teste, d_teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
