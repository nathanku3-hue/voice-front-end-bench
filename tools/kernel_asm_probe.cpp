// Disassembly probe for the sample-domain and log kernels.
//
// This translation unit exists to be compiled with `-S` and read, not to run.
// Each wrapper is thin on purpose: the kernels are `inline`, so the wrapper body
// *is* the kernel body, and the symbol name in the assembly is a stable, human
// name instead of a mangled anonymous-namespace one.
//
// The AVX2 wrappers carry the same target attribute as the kernels they wrap.
// Without it clang would be forbidden from inlining an AVX2 function into a
// baseline caller and would emit a `call` instead of the vector body, which is
// exactly the thing the audit is trying to photograph.
//
// scripts/audit_disassembly.py compiles this file twice (baseline and AVX2
// enabled), extracts each labelled block, and counts the instruction mix.

#include "fbank_kernels.h"

#include <cstddef>

extern "C" double probe_mean_scalar(const float* ring, std::size_t head, std::size_t n) {
  return voice::kernels::MeanScalarKernel(ring, head, n);
}

extern "C" void probe_preemph_scalar(const float* ring, std::size_t head, const double* window,
                                     double* out, double mean, std::size_t n) {
  voice::kernels::PreemphWindowScalarKernel(ring, head, window, out, mean, n);
}

extern "C" double probe_mean_contiguous(const float* x, std::size_t n) {
  return voice::kernels::MeanContiguousKernel(x, n);
}

__attribute__((target("avx2,fma"))) extern "C" double probe_mean_avx2(const float* x,
                                                                     std::size_t n) {
  return voice::kernels::MeanAvx2Kernel(x, n);
}

__attribute__((target("avx2,fma"))) extern "C" void probe_preemph_avx2(
    const float* x, const double* window, double* out, double mean, std::size_t n) {
  voice::kernels::PreemphWindowAvx2Kernel(x, window, out, mean, n);
}

extern "C" double probe_log_scalar(double x) { return static_cast<double>(voice::FastLog(static_cast<float>(x))); }

__attribute__((target("avx2,fma"))) extern "C" void probe_log_avx2(const float* in, double* out,
                                                                  std::size_t n) {
  voice::kernels::FastLogAvx2Kernel(in, out, n);
}
