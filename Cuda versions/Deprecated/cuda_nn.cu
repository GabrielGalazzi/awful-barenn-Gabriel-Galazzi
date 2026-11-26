// Porte funcional para CUDA de reusable_buffers_nn.cpp.
//
// Objetivo deste estágio: corretude, não velocidade. A estrutura espelha a
// versão de CPU (Forward/Backward por camada, SGD amostra a amostra), mas cada
// laço sobre neurônios ou pesos vira um kernel:
//
//   Forward  -> forward_kernel       (1 thread por neurônio de saída)
//   perda    -> loss_kernel          (1 thread; só 10 classes)
//   Backward -> delta_kernel         (1 thread por neurônio de saída)
//               grad_input_kernel    (1 thread por entrada; camadas 2 e 3)
//               update_kernel        (1 thread por peso)
//
// O que vai para a GPU uma única vez: conjunto de treino, conjunto de teste e
// pesos iniciais. Durante o treino nada trafega entre host e device; as perdas
// por época ficam na GPU e são copiadas de volta só no final.
//
// Compilação (RTX 4070 Laptop = sm_89):
//   nvcc -O3 -std=c++17 -arch=sm_89 --fmad=false cuda_nn.cu -o cuda_nn
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

constexpr int entradas = 64;
constexpr int classes = 10;

// Número de blocos necessário para cobrir n threads.
inline int blocos(int n, int threads) { return (n + threads - 1) / threads; }
// As camadas têm no máximo 32 neurônios: um warp cobre cada uma.
constexpr int threads_neuronio = 32;
constexpr int threads_peso = 256;

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

// ---------------------------------------------------------------- kernels --

// Thread i calcula o neurônio i: z = viés + produto escalar da linha i com a
// entrada. A ordem da soma (j crescente, começando pelo viés) é a mesma da CPU.
__global__ void forward_kernel(const float* __restrict__ input,
                               const float* __restrict__ weights,
                               const float* __restrict__ bias,
                               float* __restrict__ z,
                               float* __restrict__ out,
                               int in_features, int out_features, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_features) return;

    float sum = bias[i];
    const float* row = weights + i * in_features;
    for (int j = 0; j < in_features; ++j) {
        sum += input[j] * row[j];
    }
    z[i] = sum;
    // Mesma semântica de max(0.0f, sum) na CPU.
    out[i] = final ? sum : (0.0f < sum ? sum : 0.0f);
}

// Metade do erro quadrático. Uma única thread percorre as 10 classes na mesma
// ordem da CPU e acumula a perda da amostra no total da época (em double).
// Como os kernels de uma mesma stream executam em ordem, não há corrida.
__global__ void loss_kernel(const float* __restrict__ scores,
                            const int* __restrict__ label,
                            float* __restrict__ grad,
                            double* __restrict__ epoch_loss) {
    float total = 0.0f;
    int rotulo = *label;
    for (int i = 0; i < 10; ++i) {
        float erro = scores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        grad[i] = erro;
    }
    *epoch_loss += total;
}

// delta = gradiente da perda em relação a z. Na ReLU só passa quando z > 0.
__global__ void delta_kernel(const float* __restrict__ grad_out,
                             const float* __restrict__ z,
                             float* __restrict__ delta,
                             int out_features, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_features) return;
    delta[i] = final ? grad_out[i] : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Thread j soma a coluna j de W ponderada por delta. Na CPU, grad_in[j]
// acumula sobre i crescente a partir de 0; aqui a ordem é a mesma.
// Vizinhos de warp leem weights[i*in + j] e weights[i*in + j + 1]: o acesso
// é coalescido. Este kernel precisa rodar ANTES do update_kernel, porque usa
// os pesos anteriores à atualização; a ordem da stream garante isso.
__global__ void grad_input_kernel(const float* __restrict__ delta,
                                  const float* __restrict__ weights,
                                  float* __restrict__ grad_in,
                                  int in_features, int out_features) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= in_features) return;
    float acc = 0.0f;
    for (int i = 0; i < out_features; ++i) {
        acc += delta[i] * weights[i * in_features + j];
    }
    grad_in[j] = acc;
}

// Uma thread por peso. A primeira thread de cada linha também atualiza o viés.
__global__ void update_kernel(float* __restrict__ weights,
                              float* __restrict__ bias,
                              const float* __restrict__ delta,
                              const float* __restrict__ last_input,
                              float learning_rate,
                              int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= in_features * out_features) return;
    int i = idx / in_features;
    int j = idx % in_features;
    weights[idx] -= learning_rate * (delta[i] * last_input[j]);
    if (j == 0) bias[i] -= learning_rate * delta[i];
}

// ----------------------------------------------------------------- camada --

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    bool first;
    DeviceBuffer<float> weights;
    DeviceBuffer<float> bias;
    DeviceBuffer<float> last_z;
    DeviceBuffer<float> out;
    DeviceBuffer<float> delta;
    DeviceBuffer<float> grad_in;
    // Na CPU a entrada era copiada para last_input. Aqui guardamos apenas o
    // ponteiro: a entrada é a imagem (constante na GPU) ou o out da camada
    // anterior, que só é sobrescrito no Forward da próxima amostra.
    const float* last_input = nullptr;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false, bool is_first = false)
        : in_features(input_size), out_features(output_size), final(is_final), first(is_first),
          weights(size_t(output_size) * input_size), bias(output_size),
          last_z(output_size), out(output_size), delta(output_size),
          grad_in(is_first ? 0 : input_size) {
        // Pesos gerados no host com a mesma semente e a mesma distribuição da
        // CPU, depois copiados: a rede começa idêntica nas duas versões.
        vector<float> h_weights(weights.size());
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& w : h_weights) w = distribution(gen);
        weights.upload(h_weights.data(), h_weights.size());
        bias.zero();
    }

    // input é um ponteiro de device. Se destino for informado, a saída é
    // escrita lá (usado na avaliação para juntar os escores de todo o teste).
    const float* Forward(const float* input, float* destino = nullptr) {
        last_input = input;
        float* saida = destino ? destino : out.get();
        forward_kernel<<<blocos(out_features, threads_neuronio), threads_neuronio>>>(
            input, weights.get(), bias.get(), last_z.get(), saida,
            in_features, out_features, final);
        CUDA_CHECK_KERNEL();
        return saida;
    }

    const float* Backward(const float* grad_out, float learning_rate) {
        delta_kernel<<<blocos(out_features, threads_neuronio), threads_neuronio>>>(
            grad_out, last_z.get(), delta.get(), out_features, final);
        CUDA_CHECK_KERNEL();

        // Camada 1: grad_in não é usado, então não é calculado.
        if (!first) {
            grad_input_kernel<<<blocos(in_features, threads_neuronio), threads_neuronio>>>(
                delta.get(), weights.get(), grad_in.get(), in_features, out_features);
            CUDA_CHECK_KERNEL();
        }

        int n_pesos = in_features * out_features;
        update_kernel<<<blocos(n_pesos, threads_peso), threads_peso>>>(
            weights.get(), bias.get(), delta.get(), last_input, learning_rate,
            in_features, out_features);
        CUDA_CHECK_KERNEL();

        return grad_in.get();
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

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste, const DeviceDataset& d_teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Forward na GPU para cada imagem; os escores ficam juntos em
    // [n x 10] na GPU e voltam ao host em uma única cópia.
    DeviceBuffer<float> d_escores(teste.size() * classes);
    for (size_t i = 0; i < teste.size(); ++i) {
        const float* h1 = l1.Forward(d_teste.imagem(i));
        const float* h2 = l2.Forward(h1);
        l3.Forward(h2, d_escores.get() + i * classes);
    }
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
        Barenn::DenseLayer l1(Barenn::entradas, 32, false, true);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        Barenn::DeviceBuffer<float> d_gradiente(Barenn::classes);
        Barenn::DeviceBuffer<double> d_perdas(epocas);
        d_perdas.zero();

        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double* perda_epoca = d_perdas.get() + epoca;
            for (size_t k = 0; k < d_treino.n; ++k) {
                const float* h1 = l1.Forward(d_treino.imagem(k));
                const float* h2 = l2.Forward(h1);
                const float* escores = l3.Forward(h2);
                Barenn::loss_kernel<<<1, 1>>>(escores, d_treino.rotulo(k),
                                              d_gradiente.get(), perda_epoca);
                CUDA_CHECK_KERNEL();

                const float* grad3 = l3.Backward(d_gradiente.get(), taxa);
                const float* grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa);
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
        Barenn::avaliar(l1, l2, l3, teste, d_teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
