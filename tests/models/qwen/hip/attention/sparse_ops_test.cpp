// Equivalence test for the opt-in block-sparse attention.
//
// With sink + local + top blocks large enough to select every block of the
// visible context, the sparse routes must reproduce the dense matrix-core
// kernels: any difference is a selection, pooling or indexing regression, and
// it shows without a long-context run. Both carry FP16 operands and sum in a
// different order, so they agree to FP16 output precision, not exactly.
//
// Covered: prefill (16/2 and 24/4 head shapes; a chunk at depth, a ragged
// chunk, a depth that is not a multiple of the 64-key block) and decode /
// verification rows (16/2 only; one to 32 rows). Every case is also run a
// second time on warm pooled bounds, which must give the same output.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <gufo/ext/experimental/hip_utils.hpp>

#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

constexpr std::uint32_t kHeadDim = 256;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

void Fail(const char* message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

double RelativeError(const std::vector<float>& got,
                     const std::vector<float>& want) {
  double max_abs = 0.0;
  double magnitude = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i]))
      Fail("non-finite output");
    max_abs = std::fmax(max_abs, std::fabs(double(got[i]) - double(want[i])));
    magnitude = std::fmax(magnitude, std::fabs(double(want[i])));
  }
  if (!(magnitude > 0.0))
    Fail("reference output is all zero: the comparison would prove nothing");
  return max_abs / magnitude;
}

std::vector<float> Download(const float* device, std::size_t count) {
  std::vector<float> host(count);
  HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

// Every block of `context` keys fits in the selection.
gufo::hip::SparseAttentionConfig FullSelection(std::uint32_t context) {
  const std::uint32_t blocks = (context + 63U) / 64U;
  gufo::hip::SparseAttentionConfig config;
  config.sink_blocks = 2;
  config.local_blocks = 32;
  config.top_blocks = blocks;
  config.decode_top_blocks = blocks;
  return config;
}

void PrefillCase(std::uint32_t heads, std::uint32_t kv_heads,
                 std::uint32_t start_pos, std::size_t batch) {
  std::cout << "sparse prefill: heads=" << heads << "/" << kv_heads
            << " start_pos=" << start_pos << " batch=" << batch << "\n";
  const std::uint32_t context = start_pos + static_cast<std::uint32_t>(batch);
  const std::uint32_t max_context = ((context + 63U) / 64U) * 64U + 64U;
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  const std::size_t attn_width = std::size_t{heads} * kHeadDim;
  const std::size_t cache_elements = std::size_t{max_context} * kv_width;

  Rng rng(start_pos * 31U + static_cast<std::uint32_t>(batch) + heads);
  std::vector<__half> keys(cache_elements), values(cache_elements);
  for (auto& x : keys)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  for (auto& x : values)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  std::vector<float> q(batch * attn_width), gate(batch * attn_width);
  for (auto& x : q)
    x = rng.Uniform(-3.0F, 3.0F);
  for (auto& x : gate)
    x = rng.Uniform(-2.0F, 2.0F);

  gufo::test::DeviceBuffer<__half> d_k(keys), d_v(values);
  gufo::test::DeviceBuffer<float> d_q(q), d_gate(gate),
      d_dense(batch * attn_width), d_sparse(batch * attn_width),
      d_unused(kv_width);

  if (!gufo::hip::LaunchQwenWmmaAttention(
          d_q.data(), d_unused.data(), d_unused.data(), d_gate.data(), nullptr,
          nullptr, d_k.data(), d_v.data(), d_dense.data(), 0, start_pos, batch,
          max_context, heads, kv_heads, kHeadDim, nullptr, nullptr, 0,
          /*skip_kv_write=*/true))
    Fail("dense matrix-core prefill rejected a supported shape");

  const auto config = FullSelection(context);
  gufo::test::DeviceBuffer<std::uint16_t> d_pooled(
      gufo::hip::SparseAttentionPooledBytes(1, kv_heads, max_context) / 2);
  gufo::test::DeviceBuffer<std::int32_t> d_lists(
      gufo::hip::SparseAttentionListInts(batch, kv_heads, config));
  std::uint32_t pooled_end = 0;

  std::vector<float> first;
  for (int run = 0; run < 2; ++run) {
    HIP_CHECK(
        hipMemset(d_sparse.data(), 0, batch * attn_width * sizeof(float)));
    if (!gufo::hip::LaunchQwenSparseWmmaAttention(
            d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_sparse.data(),
            d_pooled.data(), d_lists.data(), &pooled_end, 0, start_pos, batch,
            max_context, heads, kv_heads, kHeadDim, config, nullptr))
      Fail("sparse prefill rejected a supported shape");
    HIP_CHECK(hipDeviceSynchronize());
    const auto got = Download(d_sparse.data(), batch * attn_width);
    const double rel =
        RelativeError(got, Download(d_dense.data(), batch * attn_width));
    std::cout << "  run " << run << ": rel=" << rel << "\n";
    if (rel > 2e-3)
      Fail("sparse prefill with every block selected differs from dense");
    if (run == 0)
      first = got;
    else if (got != first)
      Fail("sparse prefill changed between a cold and a warm bounds run");
  }

  // Control: a narrow selection over a long enough context must move the
  // output, otherwise the comparison above could not catch a regression.
  if (context >= 4096) {
    gufo::hip::SparseAttentionConfig narrow;
    narrow.sink_blocks = 1;
    narrow.local_blocks = 1;
    narrow.top_blocks = 1;
    gufo::test::DeviceBuffer<std::int32_t> d_narrow_lists(
        gufo::hip::SparseAttentionListInts(batch, kv_heads, narrow));
    std::uint32_t narrow_end = 0;
    HIP_CHECK(
        hipMemset(d_sparse.data(), 0, batch * attn_width * sizeof(float)));
    if (!gufo::hip::LaunchQwenSparseWmmaAttention(
            d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_sparse.data(),
            d_pooled.data(), d_narrow_lists.data(), &narrow_end, 0, start_pos,
            batch, max_context, heads, kv_heads, kHeadDim, narrow, nullptr))
      Fail("sparse prefill rejected a narrow selection");
    HIP_CHECK(hipDeviceSynchronize());
    const double rel =
        RelativeError(Download(d_sparse.data(), batch * attn_width),
                      Download(d_dense.data(), batch * attn_width));
    std::cout << "  narrow selection: rel=" << rel << " (must differ)\n";
    if (rel < 1e-3)
      Fail("a narrow selection left the output unchanged: the test is blind");
  }
}

void DecodeCase(std::uint32_t start_pos, std::uint32_t positions) {
  constexpr std::uint32_t kHeads = 16;
  constexpr std::uint32_t kKvHeads = 2;
  std::cout << "sparse decode: start_pos=" << start_pos << " rows=" << positions
            << "\n";
  const std::uint32_t context = start_pos + positions;
  const std::uint32_t max_context = ((context + 63U) / 64U) * 64U + 64U;
  const std::size_t kv_width = std::size_t{kKvHeads} * kHeadDim;
  const std::size_t attn_width = std::size_t{kHeads} * kHeadDim;
  const std::size_t cache_elements = std::size_t{max_context} * kv_width;

  Rng rng(start_pos * 17U + positions);
  std::vector<__half> keys(cache_elements), values(cache_elements);
  for (auto& x : keys)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  for (auto& x : values)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  std::vector<float> q(positions * attn_width), gate(positions * attn_width);
  for (auto& x : q)
    x = rng.Uniform(-3.0F, 3.0F);
  for (auto& x : gate)
    x = rng.Uniform(-2.0F, 2.0F);

  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  gufo::test::DeviceBuffer<__half> d_k(keys), d_v(values);
  gufo::test::DeviceBuffer<float> d_q(q), d_gate(gate),
      d_dense(positions * attn_width), d_sparse(positions * attn_width),
      d_scratch_dense(positions * row_scratch),
      d_scratch_sparse(positions * row_scratch);

  if (!gufo::hip::LaunchQwenWmmaDecodeAttention(
          d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_dense.data(), 0,
          start_pos, positions, max_context, kHeads, kKvHeads, kHeadDim,
          nullptr, {d_scratch_dense.data(), positions * row_scratch}))
    Fail("dense matrix-core decode rejected a supported shape");

  const auto config = FullSelection(context);
  gufo::test::DeviceBuffer<std::uint16_t> d_pooled(
      gufo::hip::SparseAttentionPooledBytes(1, kKvHeads, max_context) / 2);
  gufo::test::DeviceBuffer<std::int32_t> d_lists(
      gufo::hip::SparseAttentionListInts(positions, kKvHeads, config));
  std::uint32_t pooled_end = 0;
  std::vector<float> first;
  for (int run = 0; run < 2; ++run) {
    HIP_CHECK(
        hipMemset(d_sparse.data(), 0, positions * attn_width * sizeof(float)));
    if (!gufo::hip::LaunchQwenSparseDecodeAttention(
            d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_sparse.data(),
            d_pooled.data(), d_lists.data(), &pooled_end, 0, start_pos,
            positions, max_context, kHeads, kKvHeads, kHeadDim, config, nullptr,
            {d_scratch_sparse.data(), positions * row_scratch}))
      Fail("sparse decode rejected a supported shape");
    HIP_CHECK(hipDeviceSynchronize());
    const auto got = Download(d_sparse.data(), positions * attn_width);
    const double rel =
        RelativeError(got, Download(d_dense.data(), positions * attn_width));
    std::cout << "  run " << run << ": rel=" << rel << "\n";
    if (rel > 2e-3)
      Fail("sparse decode with every block selected differs from dense");
    if (run == 0)
      first = got;
    else if (got != first)
      Fail("sparse decode changed between a cold and a warm bounds run");
  }

  if (context >= 4096) {
    gufo::hip::SparseAttentionConfig narrow;
    narrow.sink_blocks = 1;
    narrow.local_blocks = 1;
    narrow.decode_top_blocks = 1;
    gufo::test::DeviceBuffer<std::int32_t> d_narrow_lists(
        gufo::hip::SparseAttentionListInts(positions, kKvHeads, narrow));
    std::uint32_t narrow_end = 0;
    HIP_CHECK(
        hipMemset(d_sparse.data(), 0, positions * attn_width * sizeof(float)));
    if (!gufo::hip::LaunchQwenSparseDecodeAttention(
            d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_sparse.data(),
            d_pooled.data(), d_narrow_lists.data(), &narrow_end, 0, start_pos,
            positions, max_context, kHeads, kKvHeads, kHeadDim, narrow, nullptr,
            {d_scratch_sparse.data(), positions * row_scratch}))
      Fail("sparse decode rejected a narrow selection");
    HIP_CHECK(hipDeviceSynchronize());
    const double rel =
        RelativeError(Download(d_sparse.data(), positions * attn_width),
                      Download(d_dense.data(), positions * attn_width));
    std::cout << "  narrow selection: rel=" << rel << " (must differ)\n";
    if (rel < 1e-3)
      Fail("a narrow selection left the output unchanged: the test is blind");
  }
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen block-sparse attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  for (const auto [heads, kv_heads] :
       {std::pair<std::uint32_t, std::uint32_t>{16, 2}, {24, 4}}) {
    PrefillCase(heads, kv_heads, 0, 256);
    PrefillCase(heads, kv_heads, 4096, 100);
    PrefillCase(heads, kv_heads, 4100, 333);
    PrefillCase(heads, kv_heads, 9000, 1024);
  }
  for (const std::uint32_t rows : {1U, 3U, 8U, 32U}) {
    DecodeCase(100, rows);
    DecodeCase(5000, rows);
    DecodeCase(12345, rows);
  }
  std::cout << "Qwen block-sparse attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen block-sparse attention test.\n";
  return 77;
#endif
}
