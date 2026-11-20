// IQ2_S and IQ3_XXS fractional dot products in the vendored llama.cpp tier.
//
// The device functions vec_dot_iq2_s_q8_1 and vec_dot_iq3_xxs_q8_1 apply the
// sub-block scale in integer arithmetic: IQ2_S computes
// (ls*sumi + sumi/2)/4 and IQ3_XXS (ls*sumi + sumi/2)/2, so a fractional
// contribution is truncated toward zero. llama.cpp's own CUDA/HIP kernels do
// the same; its CPU reference applies the scale in floating point. Strycast
// keeps llama.cpp's device behaviour (the G4 gate compares against it), so
// this test pins the device values and records the CPU reference beside them.
// Adopting the floating-point correction later means switching the expected
// device values to the CPU column.
//
// Inputs follow ToR-O3's 80-case sweep (gufo-org/gufo discussion #1219): two
// formats x eight 32-value sub-blocks x amplitudes {-8, -1, 0, 1, 8}. One
// weight block holds a single non-zero grid index in the chosen sub-block; the
// activation holds the amplitude at the sub-block's first position plus a
// +127/-127 pair that cancels in the dot product. Amplitude +/-1 exposes the
// truncation (IQ2_S 5.375 -> 5, IQ3_XXS 15.5 -> 15); amplitudes 0 and +/-8
// are integer-exact controls that must match the CPU reference.
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/common.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/vecdotq.hpp"
}  // namespace qfn_mmq

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace {

enum class Format { kIQ2_S, kIQ3_XXS };

struct Layout {
  const char* name;
  std::size_t block_bytes;
  std::size_t qs_stride;  ///< bytes between sub-blocks in qs
  std::uint8_t grid_index;
  double unit_value;  ///< CPU dot product at amplitude 1
};

Layout LayoutOf(Format format) {
  return format == Format::kIQ2_S ? Layout{"IQ2_S", 82, 4, 1, 43.0 / 8.0}
                                  : Layout{"IQ3_XXS", 98, 8, 5, 62.0 / 4.0};
}

std::uint16_t HalfOne() {
  return 0x3C00U;
}

std::uint16_t ToHalf(float value) {
  const __half half = __float2half(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

struct Case {
  Format format;
  int subblock;
  int amplitude;
  std::vector<std::uint8_t> weights;
  std::vector<std::uint8_t> q8_1;  ///< eight block_q8_1 (36 bytes each)
  std::vector<float> input;        ///< the same activation as 256 floats
};

Case MakeCase(Format format, int subblock, int amplitude) {
  const Layout layout = LayoutOf(format);
  Case c{format,    subblock,
         amplitude, std::vector<std::uint8_t>(layout.block_bytes, 0),
         {},        std::vector<float>(256, 0.0F)};
  const std::uint16_t d = HalfOne();
  std::memcpy(c.weights.data(), &d, sizeof(d));
  c.weights[2 + static_cast<std::size_t>(subblock) * layout.qs_stride] =
      layout.grid_index;

  c.input[subblock * 32] = static_cast<float>(amplitude);
  c.input[subblock * 32 + 30] = 127.0F;
  c.input[subblock * 32 + 31] = -127.0F;
  for (int b = 0; b < 8; ++b) {
    float sum = 0.0F;
    for (int i = 0; i < 32; ++i)
      sum += c.input[b * 32 + i];
    const std::uint16_t ds[2] = {HalfOne(), ToHalf(sum)};
    const auto* ds_bytes = reinterpret_cast<const std::uint8_t*>(ds);
    c.q8_1.insert(c.q8_1.end(), ds_bytes, ds_bytes + sizeof(ds));
    for (int i = 0; i < 32; ++i) {
      c.q8_1.push_back(static_cast<std::uint8_t>(
          static_cast<std::int8_t>(c.input[b * 32 + i])));
    }
  }
  return c;
}

// One thread walks the super-block the way mmvq does: iqs steps by the VDR
// (2) over QI (16), and each call reads Q8_1 block iqs/2.
__global__ void DotKernel(int format, const void* weights,
                          const qfn_mmq::block_q8_1* q8_1, float* out) {
  float sum = 0.0F;
  for (int iqs = 0; iqs < 16; iqs += 2) {
    sum += format == 0 ? qfn_mmq::vec_dot_iq2_s_q8_1(weights, q8_1, 0, iqs)
                       : qfn_mmq::vec_dot_iq3_xxs_q8_1(weights, q8_1, 0, iqs);
  }
  *out = sum;
}

bool Check(hipError_t error, const char* what) {
  if (error == hipSuccess)
    return true;
  std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(error));
  return false;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "no HIP device; skipping\n");
    return 77;
  }

  void* d_weights = nullptr;
  void* d_q8 = nullptr;
  float* d_out = nullptr;
  if (!Check(hipMalloc(&d_weights, 128), "hipMalloc") ||
      !Check(hipMalloc(&d_q8, 8 * 36), "hipMalloc") ||
      !Check(hipMalloc(&d_out, sizeof(float)), "hipMalloc")) {
    return 1;
  }

  int failures = 0;
  int cases = 0;
  for (const Format format : {Format::kIQ2_S, Format::kIQ3_XXS}) {
    const Layout layout = LayoutOf(format);
    for (int subblock = 0; subblock < 8; ++subblock) {
      for (const int amplitude : {-8, -1, 0, 1, 8}) {
        const Case c = MakeCase(format, subblock, amplitude);
        ++cases;

        // CPU reference: Strycast's scalar dequantizer, F64 dot product.
        std::vector<float> decoded(256);
        if (format == Format::kIQ2_S) {
          gufo::quant::DequantizeIQ2_S(c.weights.data(), decoded.data(), 256);
        } else {
          gufo::quant::DequantizeIQ3_XXS(c.weights.data(), decoded.data(), 256);
        }
        double cpu = 0.0;
        for (int i = 0; i < 256; ++i)
          cpu += static_cast<double>(decoded[i]) * c.input[i];
        const double cpu_expected = layout.unit_value * amplitude;

        // Device value, as llama.cpp's CUDA/HIP vec_dot computes it.
        const float device_expected =
            static_cast<float>(std::trunc(cpu_expected));

        float device = NAN;
        const bool ok =
            Check(hipMemset(d_weights, 0, 128), "hipMemset") &&
            Check(hipMemcpy(d_weights, c.weights.data(), c.weights.size(),
                            hipMemcpyHostToDevice),
                  "upload weights") &&
            Check(hipMemcpy(d_q8, c.q8_1.data(), c.q8_1.size(),
                            hipMemcpyHostToDevice),
                  "upload q8_1");
        if (ok) {
          DotKernel<<<1, 1>>>(format == Format::kIQ2_S ? 0 : 1, d_weights,
                              static_cast<const qfn_mmq::block_q8_1*>(d_q8),
                              d_out);
          Check(hipGetLastError(), "launch");
          Check(hipMemcpy(&device, d_out, sizeof(float), hipMemcpyDeviceToHost),
                "download");
        }

        const bool cpu_ok = cpu == cpu_expected;
        const bool device_ok = device == device_expected;
        if (!cpu_ok || !device_ok) {
          ++failures;
          std::fprintf(stderr,
                       "FAIL %s sub%d a%+d: device %.6g (want %.6g), "
                       "cpu %.6g (want %.6g)\n",
                       layout.name, subblock, amplitude, device,
                       device_expected, cpu, cpu_expected);
        }
      }
    }
  }

  hipFree(d_weights);
  hipFree(d_q8);
  hipFree(d_out);
  std::printf(
      "iq fractional dots: %d/%d cases match (device = llama.cpp HIP, "
      "cpu = float reference)\n",
      cases - failures, cases);
  return failures == 0 ? 0 : 1;
}
