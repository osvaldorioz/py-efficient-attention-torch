#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

// Función para calcular la atención eficiente
py::array_t<float> efficient_attention(const py::array_t<float>& Q,
                                        const py::array_t<float>& K,
                                        const py::array_t<float>& V) {
    auto q = Q.unchecked<2>(); // Dimensiones: [batch_size, seq_len, dim]
    auto k = K.unchecked<2>();
    auto v = V.unchecked<2>();

    int seq_len = q.shape(0);
    int dim = q.shape(1);

    // Inicializar salida
    std::vector<float> output(seq_len * dim, 0.0);

    for (int i = 0; i < seq_len; ++i) {
        std::vector<float> weights(seq_len, 0.0);
        float weight_sum = 0.0;

        // Calcular pesos (similitud coseno)
        for (int j = 0; j < seq_len; ++j) {
            float dot_product = 0.0;
            for (int d = 0; d < dim; ++d) {
                dot_product += q(i, d) * k(j, d);
            }
            weights[j] = std::exp(dot_product); // Softmax parte 1
            weight_sum += weights[j];
        }

        // Normalizar pesos y aplicar a \(V\)
        for (int j = 0; j < seq_len; ++j) {
            weights[j] /= weight_sum; // Softmax parte 2
            for (int d = 0; d < dim; ++d) {
                output[i * dim + d] += weights[j] * v(j, d);
            }
        }
    }

    // Convertir a Pybind array
    py::array_t<float> result({seq_len, dim}, output.data());
    return result;
}

// Exponer función a Python
PYBIND11_MODULE(efficient_attention_cpp, m) {
    m.def("efficient_attention", &efficient_attention, "Efficient Attention Mechanism",
          py::arg("Q"), py::arg("K"), py::arg("V"));
}
