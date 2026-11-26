// Versão otimizada do porte CUDA (cuda_nn.cu).
//
// Estágio 1: fusão de kernels. Em cuda_nn.cu cada amostra disparava 12
// kernels (3 no Forward, 1 na perda, 8 no Backward) e o treino era limitado
// pelo custo fixo de cada lançamento. Aqui uma amostra inteira (Forward das 3
// camadas, perda e Backward com atualização dos pesos) roda em um único
// kernel de um bloco. As etapas que antes eram kernels separados são
// separadas por __syncthreads(), e os valores intermediários (ativações,
// deltas e gradientes) ficam na memória compartilhada do bloco.
//
// A ordem de todas as somas é a mesma da CPU, então a saída continua bit a
// bit igual à de reusable_buffers_nn.cpp com --fmad=false.
//
// Compilação (RTX 4070 Laptop = sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false optimized_cuda_nn.cu -o optimized_cuda_nn
// --fmad=false impede que a*b+c vire uma FMA (um único arredondamento), o que
// mantém a saída bit a bit igual à da CPU. Sem a flag o resultado continua
// correto, mas os últimos dígitos da perda podem mudar.

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

// Threads do bloco que treina uma amostra. A etapa mais larga é a
// atualização dos 2048 pesos da camada 1: 8 pesos por thread.
constexpr int threads_treino = 256;
// Na avaliação cada bloco faz o Forward de uma imagem; a camada mais larga
// tem 32 neurônios, então um warp basta.
constexpr int threads_avaliacao = 32;

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

// Ponteiros de device para os parâmetros de cada camada. Os pesos de uma
// camada seguem o layout da CPU: linha i (neurônio i) em w + i * entradas.
struct Camada {
    float* w;
    float* b;
};

struct Parametros {
    Camada c1, c2, c3;
};

// ------------------------------------------------- etapas de uma camada --
// Cada função é executada por todas as threads do bloco, que dividem o
// trabalho em passos de blockDim.x. Nenhuma sincroniza: quem chama coloca
// __syncthreads() entre etapas que dependem uma da outra.

// z = viés + produto escalar da linha i com a entrada; uma thread por
// neurônio. A ordem da soma (j crescente, começando pelo viés) é a da CPU.
__device__ void forward_camada(const float* in, const Camada& c,
                               float* z, float* out,
                               int in_features, int out_features, bool final) {
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        float sum = c.b[i];
        const float* row = c.w + i * in_features;
        for (int j = 0; j < in_features; ++j) {
            sum += in[j] * row[j];
        }
        if (z) z[i] = sum;
        // Mesma semântica de max(0.0f, sum) na CPU.
        out[i] = final ? sum : (0.0f < sum ? sum : 0.0f);
    }
}

// grad_in[j] = soma, em i crescente, de delta[i] * W[i][j]; uma thread por
// entrada. Precisa rodar antes da atualização dos pesos da mesma camada.
__device__ void grad_input_camada(const float* delta, const Camada& c,
                                  float* grad_in, int in_features, int out_features) {
    for (int j = threadIdx.x; j < in_features; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < out_features; ++i) {
            acc += delta[i] * c.w[i * in_features + j];
        }
        grad_in[j] = acc;
    }
}

// Uma iteração por peso. A thread que atualiza o peso j = 0 de cada linha
// também atualiza o viés.
__device__ void update_camada(const Camada& c, const float* delta,
                              const float* last_input, float learning_rate,
                              int in_features, int out_features) {
    for (int idx = threadIdx.x; idx < in_features * out_features; idx += blockDim.x) {
        int i = idx / in_features;
        int j = idx % in_features;
        c.w[idx] -= learning_rate * (delta[i] * last_input[j]);
        if (j == 0) c.b[i] -= learning_rate * delta[i];
    }
}

// delta = gradiente da perda em relação a z de uma camada oculta (ReLU).
__device__ void delta_relu(const float* grad_out, const float* z, float* delta,
                           int out_features) {
    for (int i = threadIdx.x; i < out_features; i += blockDim.x) {
        delta[i] = grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// ---------------------------------------------------------------- kernels --

// Uma amostra de SGD inteira em um bloco. Equivale aos 12 kernels por
// amostra de cuda_nn.cu, na mesma ordem.
__global__ void train_step_kernel(Parametros p,
                                  const float* __restrict__ imagem,
                                  const int* __restrict__ rotulo,
                                  float learning_rate,
                                  double* __restrict__ epoch_loss) {
    __shared__ float x[entradas];
    __shared__ float z1[n1], h1[n1], z2[n2], h2[n2], escores[classes];
    __shared__ float d3[classes], g3[n2], d2[n2], g2[n1], d1[n1];

    for (int j = threadIdx.x; j < entradas; j += blockDim.x) x[j] = imagem[j];
    __syncthreads();

    forward_camada(x, p.c1, z1, h1, entradas, n1, false);
    __syncthreads();
    forward_camada(h1, p.c2, z2, h2, n1, n2, false);
    __syncthreads();
    forward_camada(h2, p.c3, nullptr, escores, n2, classes, true);
    __syncthreads();

    // Metade do erro quadrático. Uma única thread percorre as 10 classes na
    // mesma ordem da CPU. Na camada final delta é o próprio gradiente.
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int r = *rotulo;
        for (int i = 0; i < classes; ++i) {
            float erro = escores[i] - (i == r ? 1.0f : 0.0f);
            total += 0.5f * erro * erro;
            d3[i] = erro;
        }
        *epoch_loss += total;
    }
    __syncthreads();

    // Camada 3. grad_in usa os pesos antigos, então vem antes da atualização.
    grad_input_camada(d3, p.c3, g3, n2, classes);
    __syncthreads();
    // A atualização da camada 3 e o delta da camada 2 não dependem um do
    // outro e rodam na mesma etapa.
    update_camada(p.c3, d3, h2, learning_rate, n2, classes);
    delta_relu(g3, z2, d2, n2);
    __syncthreads();

    // Camada 2.
    grad_input_camada(d2, p.c2, g2, n1, n2);
    __syncthreads();
    update_camada(p.c2, d2, h1, learning_rate, n1, n2);
    delta_relu(g2, z1, d1, n1);
    __syncthreads();

    // Camada 1: grad_in não é usado, então não é calculado.
    update_camada(p.c1, d1, x, learning_rate, entradas, n1);
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
    forward_camada(x, p.c1, nullptr, h1, entradas, n1, false);
    __syncthreads();
    forward_camada(h1, p.c2, nullptr, h2, n1, n2, false);
    __syncthreads();
    forward_camada(h2, p.c3, nullptr, escores + size_t(blockIdx.x) * classes,
                   n2, classes, true);
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
        p.c1 = {base, base + tamanho_w1};
        p.c2 = {p.c1.b + n1, p.c1.b + n1 + tamanho_w2};
        p.c3 = {p.c2.b + n2, p.c2.b + n2 + tamanho_w3};

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
    inference_kernel<<<static_cast<unsigned>(d_teste.n), threads_avaliacao>>>(
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
            for (size_t k = 0; k < d_treino.n; ++k) {
                Barenn::train_step_kernel<<<1, Barenn::threads_treino>>>(
                    rede.p, d_treino.imagem(k), d_treino.rotulo(k), taxa, perda_epoca);
                CUDA_CHECK_KERNEL();
            }
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
