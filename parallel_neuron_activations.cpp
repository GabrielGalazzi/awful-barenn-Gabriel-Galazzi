#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

using namespace std;

// Paralelismo por neurônio (OpenMP).
//
// O treino é SGD online: cada amostra depende dos pesos atualizados pela
// anterior, então o laço de amostras continua sequencial. O que se divide
// entre as threads são os neurônios de cada camada.
//
// Uma única região paralela envolve todo o treino (em main). Dentro de
// Forward e Backward há apenas diretivas "omp for" órfãs, que dividem o
// trabalho entre a equipe já existente. Fora de uma região paralela (como em
// avaliar) essas diretivas executam em série, sem mudar nada.
//
// Toda soma de ponto flutuante mantém a ordem original das parcelas, então a
// saída é idêntica bit a bit à versão serial, com qualquer número de threads.
// Sem -fopenmp, as diretivas são ignoradas e o programa é a versão serial.

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    bool first;
    // weights[i * in_features + j] liga a entrada j ao neurônio de saída i.
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    vector<float> last_input;
    vector<float> last_z;

    vector<float> out;      // reused by Forward
    vector<float> grad_in;  // reused by Backward

    // Delta é o gradiente da perda em relação à soma antes da ativação.
    // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
    // É recalculado onde é usado para evitar uma barreira extra; a expressão
    // é a mesma da versão serial, então o valor é idêntico.
    float delta(const vector<float>& grad_out, int i) const {
        return final ? grad_out[i]
                     : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
    }

public:
    DenseLayer(int input_size, int output_size, bool is_final = false, bool is_first = false)
        : in_features(input_size), out_features(output_size), final(is_final), first(is_first),
          weights(output_size * input_size), bias(output_size, 0), last_input(input_size),
          last_z(output_size), out(output_size),
          grad_in(is_first ? 0 : input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (float& w : weights) w = distribution(gen);
    }

    // Chamado por todas as threads da equipe. Retorna o buffer compartilhado.
    const vector<float>& Forward(const float* input) {
        // Uma thread guarda a entrada para o Backward. "nowait" é seguro: a
        // barreira ao fim do laço abaixo garante que a cópia termine antes
        // de qualquer thread seguir adiante.
        #pragma omp single nowait
        copy(input, input + in_features, last_input.begin());

        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        // Neurônios são independentes: cada thread calcula um bloco deles.
        #pragma omp for schedule(static)
        for (int i = 0; i < out_features; ++i) {
            float sum = bias[i];
            const float* row = &weights[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                sum += input[j] * row[j];
            }
            last_z[i] = sum;
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            out[i] = final ? sum : max(0.0f, sum);
        }
        // Barreira implícita: "out" está completo para todas as threads.
        return out;
    }

    // Chamado por todas as threads da equipe. Retorna o buffer compartilhado.
    const vector<float>& Backward(const vector<float>& grad_out, float learning_rate) {
        if (!first) {
            // Passo 1: gradiente da entrada, dividido por j.
            // grad_in[j] soma delta[i] * w[i][j] com i em ordem crescente,
            // a mesma ordem da versão serial, então o resultado é idêntico.
            // Usa os pesos anteriores à atualização.
            #pragma omp for schedule(static)
            for (int j = 0; j < in_features; ++j) {
                float g = 0.0f;
                for (int i = 0; i < out_features; ++i) {
                    g += delta(grad_out, i) * weights[i * in_features + j];
                }
                grad_in[j] = g;
            }
            // Barreira implícita: nenhuma thread atualiza pesos antes de
            // todas terminarem de lê-los no passo 1.
        }

        // Passo 2: atualização de pesos e viés, dividida por i.
        // Cada linha de pesos pertence a um único neurônio, então não há
        // conflito de escrita entre threads.
        #pragma omp for schedule(static)
        for (int i = 0; i < out_features; ++i) {
            const float d = delta(grad_out, i);
            float* row = &weights[i * in_features];
            for (int j = 0; j < in_features; ++j) {
                row[j] -= learning_rate * (d * last_input[j]);
            }
            bias[i] -= learning_rate * d;
        }
        // Barreira implícita: pesos atualizados antes do próximo uso.

        return grad_in;
    }
};

//Todos os dados ficam armazenados em um espaço contínuo de memória agora.
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
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem[lidos++] = valor / 16.0f;
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
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

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

// Roda fora de qualquer região paralela: as diretivas das camadas executam
// em série aqui.
void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& h1 = l1.Forward(amostra.imagem.data());
        const auto& h2 = l2.Forward(h1.data());
        const auto& escores = l3.Forward(h2.data());
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
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
    // Ordena os pares incorretos do mais frequente para o menos frequente.
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
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32, false, true);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};
        // Compartilhada pela equipe; só é alterada dentro de "omp single".
        double perda_total = 0;

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();

        // Uma única região paralela para todo o treino: as threads são
        // criadas uma vez, e não a cada camada de cada amostra. Todas as
        // threads percorrem os mesmos laços de época e amostra; o trabalho
        // dentro das camadas é dividido pelas diretivas "omp for".
        #pragma omp parallel
        {
            for (int epoca = 0; epoca < epocas; ++epoca) {
                // Uma época percorre todo o conjunto na ordem do arquivo.
                for (const auto& amostra : treino) {
                    const auto& h1 = l1.Forward(amostra.imagem.data());
                    const auto& h2 = l2.Forward(h1.data());
                    const auto& escores = l3.Forward(h2.data());

                    // Uma só thread calcula a perda e o gradiente de saída;
                    // a barreira implícita os publica para as demais.
                    #pragma omp single
                    perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                    // Propaga o gradiente da última camada até a primeira.
                    // Cada Backward também atualiza os parâmetros da sua camada.
                    const auto& grad3 = l3.Backward(gradiente, taxa);
                    const auto& grad2 = l2.Backward(grad3, taxa);
                    l1.Backward(grad2, taxa);
                }

                #pragma omp single
                {
                    perdas[epoca] = perda_total / treino.size();
                    perda_total = 0;
                }
            }
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
