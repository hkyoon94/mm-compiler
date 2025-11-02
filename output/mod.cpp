// This code is auto-generated.
#include <torch/extension.h>


// Compiled C++ code of legalized op: 'mm_naive'
void mm_naive(const float* A, const float* B, float* C) {
  for (size_t m = 0; m < 1024; ++m) {
    for (size_t n = 0; n < 1024; ++n) {
      for (size_t k = 0; k < 1024; ++k) {  // reduction over k
        float a_1 = A[(m) * (1024) + k];
        float b_1 = B[(k) * (1024) + n];
        float mul_1 = a_1 * b_1;
        C[(m) * (1024) + n] += mul_1;
      }
    }
  }
}

torch::Tensor c_reduction_wrapper_mm_naive(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension mismatch");

  const auto M = A.size(0);
  const auto N = B.size(1);
  auto C = torch::zeros({M, N}, torch::kFloat);

  mm_naive(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}

// Compiled C++ code of legalized op: 'mm_opt'
void mm_opt(const float* A, const float* B, float* C) {
  for (size_t m_blk = 0; m_blk < 4; ++m_blk) {
    for (size_t n_blk = 0; n_blk < 4; ++n_blk) {
      for (size_t k_blk = 0; k_blk < 4; ++k_blk) {
        for (size_t m_inner_blk = 0; m_inner_blk < 8; ++m_inner_blk) {
          for (size_t n_inner_blk = 0; n_inner_blk < 2; ++n_inner_blk) {
            for (size_t k_inner_blk = 0; k_inner_blk < 8; ++k_inner_blk) {
              for (size_t m_inner_inner = 0; m_inner_inner < 32; ++m_inner_inner) {
                for (size_t k_inner_inner = 0; k_inner_inner < 32; ++k_inner_inner) {  // reduction over k_inner_inner
                  float a_1 = A[((m_blk * 256 + (m_inner_blk * 32 + m_inner_inner))) * (1024) + (k_blk * 256 + (k_inner_blk * 32 + k_inner_inner))];   // hoisted
                  for (size_t n_inner_inner = 0; n_inner_inner < 128; ++n_inner_inner) {
                    float b_1 = B[((k_blk * 256 + (k_inner_blk * 32 + k_inner_inner))) * (1024) + (n_blk * 256 + (n_inner_blk * 128 + n_inner_inner))];
                    float mul_1 = a_1 * b_1;
                    C[((m_blk * 256 + (m_inner_blk * 32 + m_inner_inner))) * (1024) + (n_blk * 256 + (n_inner_blk * 128 + n_inner_inner))] += mul_1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

torch::Tensor c_reduction_wrapper_mm_opt(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension mismatch");

  const auto M = A.size(0);
  const auto N = B.size(1);
  auto C = torch::zeros({M, N}, torch::kFloat);

  mm_opt(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
  return C;
}


PYBIND11_MODULE(mod, m) {
  m.def("mm_naive", &c_reduction_wrapper_mm_naive, "");
  m.def("mm_opt", &c_reduction_wrapper_mm_opt, "");
}
