// Block-sparse masked prefill attention for the dense-attention Qwen models
// (training-free, opt-in through STRYCAST_SPARSE_ATTN). A query tile of 32
// queries attends to its sink blocks, a local window and the highest scoring
// older 64-token blocks, so attention cost no longer grows with depth.
//
// Block scores follow Quest: every 64-key block keeps the per-channel minimum
// and maximum of its keys, and sum over channels of max(q * min, q * max)
// bounds the block's best dot product. Queries of a tile share one selection,
// scored with the tile's mean query per head, summed over the heads of a KV
// group. The attention kernel below is the dense WMMA kernel of
// attention_wmma.hip with its key loop driven by the selected block list.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <span>
#include <type_traits>

#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/kv_q8.hpp"
#include "src/families/qwen/kernels/ops/attention.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kHeadDim = detail::kTiledAttentionHeadDim;
constexpr std::uint32_t kSparseBlockKeys = 64;
constexpr std::uint32_t kSelectMaxBlocks = 4096;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

constexpr std::uint32_t kWmmaHeads = 2;
constexpr std::uint32_t kWmmaQueryRows = 32;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaKSteps = kHeadDim / 16;
constexpr std::uint32_t kWmmaKStride = kHeadDim + 8;

}  // namespace

template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kHeadMajor,
         std::uint32_t kQueryHeads, std::uint32_t kKvHeads>
__launch_bounds__(256, 2) __global__
    void SparseWmmaCausalAttention(const float* __restrict__ q,
                             const float* __restrict__ gate,
                             const __half* __restrict__ k_cache_f16,
                             const __half* __restrict__ v_cache_f16,
                             float* __restrict__ out_context,
                             std::uint32_t start_pos, std::size_t batch_size,
                             const std::int32_t* __restrict__ block_lists,
                             std::uint32_t list_stride) {
  constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
  constexpr std::uint32_t kAttnWidth = kQueryHeads * kHeadDim;
  constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
  static_assert(kGqa % kWmmaHeads == 0, "head pairs must divide the GQA group");
  constexpr std::uint32_t kRowBlocks = (kQueryRows / 16) * kWmmaHeads;
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / (8 / kSTiles);
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == 4, "eight waves cover four S tiles in two halves");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && kQueryRows % 16 == 0, "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 8 / kSTiles == kWmmaKSteps, "k split");
  static_assert(!kHeadMajor || kKeys == 16, "packed value tile width");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  const std::uint32_t query_start = blockIdx.x * kQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / kWmmaHeads);
  const std::uint32_t pair_in_group = head_pair % (kGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row block -> (query head, 16-query sub-block of this block's rows).
  const auto row_block_head = [&](std::uint32_t rb) {
    return first_query_head + (rb % kWmmaHeads);
  };
  const auto row_block_offset = [&](std::uint32_t rb) {
    return (rb / kWmmaHeads) * 16;
  };

  constexpr std::uint32_t kKvLdsHalves =
      kHeadMajor || kKeys * kWmmaKStride > kHeadDim * kVtStride
          ? kKeys * kWmmaKStride
          : kHeadDim * kVtStride;
  __shared__ __half kv_lds[kKvLdsHalves];
  // Padding avoids repeated banks in softmax reads and probability fragments.
  __shared__ float s_lds[2][kSTiles][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t local_query =
        query_start + row_block_offset(s_rb) + sub;
    const bool live = local_query < batch_size;
    const float* q_row =
        q + (static_cast<std::size_t>(local_query) * kAttnWidth) +
        (static_cast<std::size_t>(row_block_head(s_rb)) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block, so both
  // halves of the head dimension stay evenly loaded.
  v8f o_acc[kRowBlocks][2] = {};
  // Each softmax row keeps the same four owner lanes across the key sweep.
  // Only the final statistics and each tile's rescale factor cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  // The selected 64-token blocks of this query tile and KV head, ascending.
  // Entry 0 is the count; each block is four 16-key tiles.
  const std::int32_t* list =
      block_lists + ((static_cast<std::size_t>(blockIdx.x) * kKvHeads +
                      kv_head) *
                     list_stride);
  const std::uint32_t tile_count = static_cast<std::uint32_t>(list[0]) * 4u;
  const auto tile_key = [&](std::uint32_t t) {
    return (static_cast<std::uint32_t>(list[1 + (t >> 2)]) * kSparseBlockKeys) +
           ((t & 3u) * kKeys);
  };

  // One key per lane for V, a 16-dim slice per thread. This costs a strided
  // global read but makes the transpose's LDS writes conflict-free, which is
  // worth 1.8x; a coalesced remapping was measured and is slower at depth.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  constexpr std::size_t kv_stride = kHeadMajor ? kHeadDim : kKvWidth;
  const std::size_t head_offset = static_cast<std::size_t>(kv_head) * kHeadDim *
                                  (kHeadMajor ? context_end : 1);
  const auto* v_base = v_cache_f16 + head_offset + v_slice;
  const auto* k_base = k_cache_f16 + head_offset;

  using ValueFragment = std::conditional_t<kHeadMajor, v16h, uint4>;
  constexpr std::uint32_t kValueFragments = kHeadMajor ? 2 : kVRegs;
  const auto load_v = [&](std::uint32_t key_start, ValueFragment* dst) {
    if constexpr (kHeadMajor) {
      // Each wave consumes two dimension tiles. The temporary value layout
      // supplies their WMMA fragments directly, avoiding a repeated LDS
      // transpose and barrier for every 16-key tile.
      const std::size_t padded = (context_end + 15u) & ~std::size_t{15};
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
        const std::uint32_t dim = (wave + t * 8u) * 16u + sub;
        dst[t] = LoadFrag(
            v_cache_f16 +
            static_cast<std::size_t>(kv_head) * padded * kHeadDim +
            static_cast<std::size_t>(key_start / 16u) * kHeadDim * 16u +
            dim * 16u);
      }
    } else {
      const std::uint32_t key_position = key_start + v_key;
      const auto* src =
          v_base + (static_cast<std::size_t>(key_position) * kv_stride);
      const bool live = key_position < context_end;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                      : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](std::uint32_t key_start, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = key_start + (idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      dst[n] =
          (key_position < context_end)
              ? *reinterpret_cast<const uint4*>(
                    k_base +
                    (static_cast<std::size_t>(key_position) * kv_stride) + d8)
              : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  uint4 k_cur[kKRegs];
  ValueFragment v_cur[kValueFragments];
  uint4 k_pre[kKRegs] = {};
  if (tile_count != 0) {
    load_k(tile_key(0), k_cur);
    load_v(tile_key(0), v_cur);
  }

  for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
    const std::uint32_t key_start = tile_key(tile);
    // --- stage K from the registers the previous iteration prefetched
    __syncthreads();
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&kv_lds[(key_row * kWmmaKStride) + d8]) =
          k_cur[n];
    }
    __syncthreads();

    // --- prefetch the next key tile. Everything below covers its latency.
    const bool has_next = tile + 1 < tile_count;
    if (has_next) {
      const std::uint32_t next_start = tile_key(tile + 1);
      load_k(next_start, k_pre);
    }

    // --- S = Q K^T ---
    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&kv_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
      }
      // Each half writes its own slot, so the reader sums them instead of a
      // read-modify-write that would need a barrier between the halves.
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
    __syncthreads();

    // --- online softmax: kSoftmaxLanes threads per query row ---
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query =
          query_start + row_block_offset(rb) + row;
      const std::uint32_t absolute_query = start_pos + local_query;
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = key_start + col;
        const bool valid = local_query < batch_size &&
                           key_position <= absolute_query &&
                           key_position < context_end;
        const std::uint32_t tile = ((col / 16) * kRowBlocks) + rb;
        vals[m] = valid ? (s_lds[0][tile][row][col % 16] +
                           s_lds[1][tile][row][col % 16])
                        : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][(seg * kPerLane) + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += __shfl_xor(part_sum, off);
      }
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

    // The unpacked path shares each scale across both dimension tiles. Packed
    // V instead rescales immediately before PV, shortening register lifetimes
    // enough to retain six waves per SIMD with the causal-tail correction.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        float scale[8];
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
        }
#pragma unroll
        for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            o_acc[rb][t][i] *= scale[i];
          }
        }
      }
    }

    // --- stage V transposed. No barrier is needed here: the barrier after the
    // softmax already separates every read of K in the S phase from these
    // writes to the same buffer.
    //
    // Lane L owns key L % kKeys and each thread a dim slice, so consecutive
    // lanes write consecutive *columns* of V^T: two 16-bit writes share a dword
    // and the wave covers 16 banks. The mapping the coalesced global read would
    // prefer -- each lane taking 8 contiguous dims of one key -- puts all 32
    // lanes on rows kVtStride halves apart, which is one bank, and costs a
    // 32-way conflict on every write.
    if constexpr (!kHeadMajor) {
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_cur[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          kv_lds[((v_slice + (j * 8) + i) * kVtStride) + v_key] = packed[i];
        }
      }
      // v_cur is dead once staged: fetch the next tile's V into it now. The PV
      // work below, the next tile's QK and softmax cover the latency, and no
      // second set of V registers stays live across the sweep.
      if (has_next)
        load_v(tile_key(tile + 1), v_cur);
      __syncthreads();
    }

    // --- O += P V ---
    // Packed V is in registers: visit each row block's two dimension tiles
    // together, sharing its scales. Unpacked V comes from LDS: visit dimension
    // tiles first so the compiler can share their fragments across row blocks.
    // The order is compile-time constant and both paths keep the same
    // arithmetic.
#pragma unroll
    for (std::uint32_t outer = 0; outer < (kHeadMajor ? kRowBlocks : 2);
         ++outer) {
      float scale[8];
      if constexpr (kHeadMajor) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[outer * 16 + 2 * i + half_id];
      }
#pragma unroll
      for (std::uint32_t inner = 0; inner < (kHeadMajor ? 2 : kRowBlocks);
           ++inner) {
        const std::uint32_t rb = kHeadMajor ? outer : inner;
        const std::uint32_t t = kHeadMajor ? inner : outer;
        const std::uint32_t dim_tile = wave + t * 8;
        // Keep each (row block, dim tile) step's fragment loads next to its
        // WMMA: left alone, the scheduler hoists every step's loads together.
        __builtin_amdgcn_sched_barrier(0);
        v16h v_frag[kKeyBlocks];
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          if constexpr (kHeadMajor)
            v_frag[kb] = v_cur[t];
          else
            v_frag[kb] =
                LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride + kb * 16]);
        }
        if constexpr (kHeadMajor) {
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i)
            o_acc[rb][t][i] *= scale[i];
        }
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          const auto first_key = key_start + kb * 16;
          // Like Flash-Next, sum a query's final partial tile over visible
          // keys only. WMMA can round differently when a zero probability
          // multiplies a populated future value instead of chunk padding.
          // Complete tiles keep the matrix-core path and its original order.
          if (first_key + 15 > start_pos + query_start) {
            v8f tail = o_acc[rb][t];
            // Keep the position difference uniform across the wave; only the
            // even/odd row owner varies between its two half-waves.
            const int relative_key = static_cast<int>(first_key) -
                                     static_cast<int>(start_pos + query_start +
                                                      row_block_offset(rb));
            // Not unrolled: with 16 keys x 8 rows unrolled, the compiler hoists
            // the V converts and spills the accumulators (60 B of scratch with
            // the CI compiler, HIP Clang 24). The unpacked path reads V back
            // from LDS, the values v_frag holds, so the key loop stays a loop.
#pragma unroll 1
            for (std::uint32_t key = 0; key < 16; ++key) {
              float value;
              if constexpr (kHeadMajor)
                value = static_cast<float>(v_frag[kb][key]);
              else
                value = __half2float(
                    kv_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16) +
                           key]);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                if (static_cast<int>(half_id) <
                        relative_key + 15 - static_cast<int>(2 * i) &&
                    static_cast<int>(half_id) >= relative_key +
                                                     static_cast<int>(key) -
                                                     static_cast<int>(2 * i)) {
                  tail[i] =
                      fmaf(__half2float(
                               p_lds[rb * 16 + 2 * i + half_id][kb * 16 + key]),
                           value, tail[i]);
                }
              }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i)
              if (static_cast<int>(half_id) <
                  relative_key + 15 - static_cast<int>(2 * i))
                next[i] = tail[i];
          }
          o_acc[rb][t] = next;
        }
      }
    }

#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      k_cur[n] = k_pre[n];
    }
    if constexpr (kHeadMajor) {
      // Packed V is already in WMMA order. Load it after the current PV work;
      // the next QK/softmax hides its latency without retaining a second set
      // of fragments across the causal-tail calculation.
      if (has_next)
        load_v(tile_key(tile + 1), v_cur);
    }
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  // --- epilogue ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
    const std::uint32_t query_head = row_block_head(rb);
    const std::uint32_t row_offset = row_block_offset(rb);
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        const std::uint32_t local_query = query_start + row_offset + row;
        if (local_query >= batch_size) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(local_query) * kAttnWidth) +
            (static_cast<std::size_t>(query_head) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= 1.0F / (1.0F + __expf(-gate[offset]));
        }
        out_context[offset] = value;
      }
    }
  }
}

namespace {

// Min and max of every key channel for the 64-key blocks [first_block, ...)
// that end at or before `end_pos` keys (the last block may be partial).
__global__ void UpdatePooledKeys(const __half* __restrict__ k_layer,
                                 __half* __restrict__ pooled,
                                 std::uint32_t first_block,
                                 std::uint32_t end_pos,
                                 std::uint32_t num_kv_heads,
                                 std::uint32_t max_blocks) {
  const std::uint32_t block = first_block + blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t dim = threadIdx.x;
  float lo = INFINITY;
  float hi = -INFINITY;
  for (std::uint32_t token = 0; token < kSparseBlockKeys; ++token) {
    const std::uint32_t position = (block * kSparseBlockKeys) + token;
    if (position >= end_pos)
      break;
    const float value = __half2float(
        k_layer[((static_cast<std::size_t>(position) * num_kv_heads) + head) *
                    kHeadDim +
                dim]);
    lo = fminf(lo, value);
    hi = fmaxf(hi, value);
  }
  __half* out =
      pooled + ((static_cast<std::size_t>(head) * max_blocks) + block) * 512;
  out[dim] = __float2half(lo);
  out[256 + dim] = __float2half(hi);
}

template<std::uint32_t kQueryHeads, std::uint32_t kKvHeads>
__launch_bounds__(256) __global__ void SelectSparseBlocks(
    const float* __restrict__ q, const __half* __restrict__ pooled,
    std::int32_t* __restrict__ lists, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_blocks, std::uint32_t sink_blocks,
    std::uint32_t local_blocks, std::uint32_t top_blocks,
    std::uint32_t list_stride) {
  constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
  __shared__ float score[kSelectMaxBlocks];
  __shared__ std::uint32_t mark[kSelectMaxBlocks / 32];
  __shared__ float qmean[kGqa][kHeadDim];
  __shared__ float red_value[8];
  __shared__ int red_index[8];
  __shared__ std::uint32_t counts[256];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t tile = blockIdx.x;
  const std::uint32_t kv_head = blockIdx.y;
  const std::uint32_t first_q = tile * kWmmaQueryRows;
  const std::uint32_t n_q = static_cast<std::uint32_t>(
      min(static_cast<std::size_t>(kWmmaQueryRows), batch_size - first_q));
  const std::uint32_t last_block = (start_pos + first_q + n_q - 1) / kSparseBlockKeys;

  for (std::uint32_t idx = tid; idx < kGqa * kHeadDim; idx += 256) {
    const std::uint32_t h = idx / kHeadDim;
    const std::uint32_t d = idx % kHeadDim;
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < n_q; ++i) {
      sum += q[((static_cast<std::size_t>(first_q + i) * kQueryHeads) +
                (kv_head * kGqa) + h) *
                   kHeadDim +
               d];
    }
    qmean[h][d] = sum / static_cast<float>(n_q);
  }
  for (std::uint32_t i = tid; i < kSelectMaxBlocks / 32; i += 256)
    mark[i] = 0;
  __syncthreads();

  const std::uint32_t visible_blocks = last_block + 1;
  const std::uint32_t local_start =
      max(sink_blocks, visible_blocks > local_blocks
                           ? visible_blocks - local_blocks
                           : 0u);
  const std::uint32_t cand_begin = sink_blocks;
  const std::uint32_t cand_end = max(local_start, cand_begin);
  const std::uint32_t n_cand =
      cand_end > cand_begin && visible_blocks > cand_begin
          ? min(cand_end, visible_blocks) - cand_begin
          : 0u;

  float qr[kGqa][8];
#pragma unroll
  for (std::uint32_t h = 0; h < kGqa; ++h)
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j)
      qr[h][j] = qmean[h][(lane * 8) + j];

  for (std::uint32_t b = wave; b < n_cand; b += 8) {
    const __half* base =
        pooled +
        ((static_cast<std::size_t>(kv_head) * max_blocks) + cand_begin + b) *
            512;
    union {
      uint4 raw;
      __half h[8];
    } lo, hi;
    lo.raw = *reinterpret_cast<const uint4*>(base + (lane * 8));
    hi.raw = *reinterpret_cast<const uint4*>(base + 256 + (lane * 8));
    float part = -INFINITY;
#pragma unroll
    for (std::uint32_t h = 0; h < kGqa; ++h) {
      float head_part = 0.0F;
#pragma unroll
      for (std::uint32_t j = 0; j < 8; ++j)
        head_part += fmaxf(qr[h][j] * __half2float(lo.h[j]),
                           qr[h][j] * __half2float(hi.h[j]));
#pragma unroll
      for (std::uint32_t off = 16; off > 0; off >>= 1)
        head_part += __shfl_xor(head_part, off);
      part = fmaxf(part, head_part);
    }
    if (lane == 0)
      score[b] = part;
  }
  __syncthreads();

  const std::uint32_t picks = min(top_blocks, n_cand);
  if (picks >= n_cand) {
    for (std::uint32_t i = tid; i < n_cand; i += 256) {
      const std::uint32_t block = cand_begin + i;
      atomicOr(&mark[block >> 5], 1u << (block & 31u));
    }
  } else if (picks > 0) {
    // Largest ordered-key threshold that still keeps at least `picks` blocks.
    const auto ordered = [](float v) {
      const std::uint32_t bits = __float_as_uint(v);
      return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
    };
    __shared__ std::uint32_t red_count[8];
    std::uint32_t lo = 0;
    std::uint32_t hi = 0xFFFFFFFFu;
    while (lo < hi) {
      const std::uint32_t mid = lo + static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(hi) - lo + 1u) >> 1u);
      std::uint32_t count = 0;
      for (std::uint32_t i = tid; i < n_cand; i += 256)
        count += ordered(score[i]) >= mid ? 1u : 0u;
#pragma unroll
      for (std::uint32_t off = 16; off > 0; off >>= 1)
        count += __shfl_xor(count, off);
      if (lane == 0)
        red_count[wave] = count;
      __syncthreads();
      std::uint32_t total = 0;
      for (std::uint32_t w = 0; w < 8; ++w)
        total += red_count[w];
      __syncthreads();
      if (total >= picks)
        lo = mid;
      else
        hi = mid - 1u;
    }
    for (std::uint32_t i = tid; i < n_cand; i += 256) {
      if (ordered(score[i]) > lo) {
        const std::uint32_t block = cand_begin + i;
        atomicOr(&mark[block >> 5], 1u << (block & 31u));
      }
    }
    __syncthreads();
    if (tid == 0) {
      std::uint32_t kept = 0;
      for (std::uint32_t i = 0; i < n_cand; ++i)
        kept += ordered(score[i]) > lo ? 1u : 0u;
      for (std::uint32_t i = 0; i < n_cand && kept < picks; ++i) {
        if (ordered(score[i]) == lo) {
          const std::uint32_t block = cand_begin + i;
          mark[block >> 5] |= 1u << (block & 31u);
          ++kept;
        }
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    for (std::uint32_t b = 0; b < min(sink_blocks, visible_blocks); ++b)
      mark[b >> 5] |= 1u << (b & 31u);
    for (std::uint32_t b = local_start; b < visible_blocks; ++b)
      mark[b >> 5] |= 1u << (b & 31u);
  }
  __syncthreads();

  // Each thread owns 16 consecutive block ids; an exclusive scan of their
  // set-bit counts gives every selected block its place in the ascending list.
  const std::uint32_t word = mark[tid >> 1] >> ((tid & 1u) * 16u);
  const std::uint32_t bits = word & 0xFFFFu;
  counts[tid] = __popc(bits);
  __syncthreads();
  if (tid == 0) {
    std::uint32_t total = 0;
    for (std::uint32_t i = 0; i < 256; ++i) {
      const std::uint32_t c = counts[i];
      counts[i] = total;
      total += c;
    }
    lists[(static_cast<std::size_t>(tile) * kKvHeads + kv_head) *
          list_stride] = static_cast<std::int32_t>(total);
  }
  __syncthreads();
  std::int32_t* out =
      lists + ((static_cast<std::size_t>(tile) * kKvHeads + kv_head) *
               list_stride) + 1 + counts[tid];
  std::uint32_t remaining = bits;
  std::uint32_t written = 0;
  while (remaining != 0) {
    const std::uint32_t bit = __ffs(remaining) - 1;
    remaining &= remaining - 1;
    out[written++] = static_cast<std::int32_t>((tid * 16u) + bit);
  }
}

}  // namespace

namespace {
namespace dec {

// Decode and verification over the selected blocks: the matrix-core decode
// kernel of attention_wmma_decode.hip with its key loop driven by the block
// list (eight query heads per 256-wide KV head, up to kPositions positions
// per launch as a 64-row query tile).
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGroup = 8;
#ifndef GUFO_WMMA_DECODE_POSITIONS
#define GUFO_WMMA_DECODE_POSITIONS 4
#endif
constexpr std::uint32_t kPositions = GUFO_WMMA_DECODE_POSITIONS;
constexpr std::uint32_t kRows = kGroup * kPositions;
constexpr std::uint32_t kRowBlocks = kRows / 16;
constexpr std::uint32_t kDimSplits = 8 / kRowBlocks;
constexpr std::uint32_t kKeys = 16;
constexpr std::uint32_t kKSteps = kHeadDim / 16;
constexpr std::uint32_t kKStepsPerWave = kKSteps / kDimSplits;
constexpr std::uint32_t kKStride = kHeadDim + 8;
constexpr std::uint32_t kVtStride = kKeys + 8;
constexpr std::uint32_t kSoftmaxLanes = 256 / kRows;
constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
constexpr std::uint32_t kMinPartition = 256;
static_assert(kRowBlocks * kDimSplits == 8 && kKRegs == 2 && kVRegs == 2 &&
              kKeys % kSoftmaxLanes == 0);

template<bool kQ8>
__launch_bounds__(256, 2) __global__ void SparseDecodePartials(
    const float* __restrict__ q, const void* __restrict__ k_layer,
    const void* __restrict__ v_layer, float* __restrict__ scratch,
    std::uint32_t start_pos, std::uint32_t positions,
    const std::int32_t* __restrict__ block_lists, std::uint32_t list_stride,
    std::uint32_t num_heads, std::uint32_t num_kv_heads) {
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;
  const std::uint32_t split = blockIdx.x;
  const std::uint32_t kv_head = blockIdx.y;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * kHeadDim;
  const std::size_t attn_width = static_cast<std::size_t>(num_heads) * kHeadDim;
  constexpr float attention_scale = 1.0F / 16.0F;

  // Keys are visited through the selected 64-key blocks: slot s is key
  // list[1 + s / 64] * 64 + s % 64. A partition is a run of slots, so a 16-key
  // tile is always contiguous in the cache.
  const std::uint32_t key_end = start_pos + positions;
  const std::int32_t* list = block_lists + kv_head * list_stride;
  const std::uint32_t slots = static_cast<std::uint32_t>(list[0]) * 64u;
  constexpr std::uint32_t max_splits_ =
      detail::kSplitKDecodeAttentionMaxSplits;
  const std::uint32_t per_split =
      max(kMinPartition, ((slots + max_splits_ - 1) / max_splits_ + 15u) / 16u * 16u);
  const std::uint32_t slot_begin = min(split * per_split, slots);
  const std::uint32_t slot_end = min(slot_begin + per_split, slots);
  const auto key_of = [&](std::uint32_t slot) {
    return static_cast<std::uint32_t>(list[1 + slot / 64u]) * 64u + slot % 64u;
  };
  const auto row_position = [](std::uint32_t row) { return row / kGroup; };
  const auto row_head = [&](std::uint32_t row) {
    return kv_head * kGroup + (row % kGroup);
  };

  __shared__ __half kv_lds[kHeadDim * kVtStride];
  __shared__ float s_lds[kDimSplits][kRowBlocks][16][17];
  __shared__ __half p_lds[kRows][kKeys + 8];
  __shared__ float row_max[kRows];
  __shared__ float row_sum[kRows];
  __shared__ float row_scale[kRows];
  static_assert(kHeadDim * kVtStride >= kKeys * kKStride);

  // Wave w computes the score tile of row block w % kRowBlocks over part
  // w / kRowBlocks of the head dimension.
  const std::uint32_t s_rb = wave % kRowBlocks;
  const std::uint32_t s_kh = wave / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  {
    const std::uint32_t row = s_rb * 16 + sub;
    const bool live = row_position(row) < positions;
    const float* q_row =
        q + static_cast<std::size_t>(row_position(row)) * attn_width +
        static_cast<std::size_t>(row_head(row)) * kHeadDim;
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = static_cast<_Float16>(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = static_cast<_Float16>(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = static_cast<_Float16>(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = static_cast<_Float16>(f.w * attention_scale);
      }
    }
  }

  // This wave owns dimension tiles `wave` and `wave + 8` of every row block.
  v8f o_acc[kRowBlocks][2] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* k_base = static_cast<const __half*>(k_layer) +
                       static_cast<std::size_t>(kv_head) * kHeadDim;
  const auto* v_base = static_cast<const __half*>(v_layer) +
                       static_cast<std::size_t>(kv_head) * kHeadDim + v_slice;
  // 8-bit rows: thread tid stages 16 dimensions of key tid / 16.
  const std::size_t q8_stride =
      static_cast<std::size_t>(num_kv_heads) * kv_q8::kRowBytes;
  const auto* k_rows = static_cast<const std::uint8_t*>(k_layer) +
                       static_cast<std::size_t>(kv_head) * kv_q8::kRowBytes;
  const auto* v_rows = static_cast<const std::uint8_t*>(v_layer) +
                       static_cast<std::size_t>(kv_head) * kv_q8::kRowBytes;
  const std::uint32_t q8_key = tid / 16u;
  const std::uint32_t q8_dim = (tid % 16u) * 16u;
  using KRegs = std::conditional_t<kQ8, kv_q8::Slice16, uint4[kKRegs]>;
  using VRegs = std::conditional_t<kQ8, kv_q8::Slice16, uint4[kVRegs]>;
  // Keys at or past key_end are never read: positions past the last row may
  // not be written yet.
  const auto load_v = [&](std::uint32_t key_start, VRegs& dst) {
    const std::uint32_t key = key_start + v_key;
    if constexpr (kQ8) {
      dst = key < key_end
                ? kv_q8::LoadSlice16(v_rows + key * q8_stride, v_slice)
                : kv_q8::Slice16{make_uint4(0u, 0u, 0u, 0u), 0.0F};
    } else {
      const auto* src = v_base + static_cast<std::size_t>(key) * kv_width;
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j)
        dst[j] = key < key_end ? *reinterpret_cast<const uint4*>(src + j * 8)
                               : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  const auto load_k = [&](std::uint32_t key_start, KRegs& dst) {
    if constexpr (kQ8) {
      const std::uint32_t key = key_start + q8_key;
      dst = key < key_end
                ? kv_q8::LoadSlice16(k_rows + key * q8_stride, q8_dim)
                : kv_q8::Slice16{make_uint4(0u, 0u, 0u, 0u), 0.0F};
    } else {
#pragma unroll
      for (std::uint32_t n = 0; n < kKRegs; ++n) {
        const std::uint32_t idx = tid + n * 256;
        const std::uint32_t key = key_start + idx / (kHeadDim / 8);
        const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
        dst[n] = key < key_end ? *reinterpret_cast<const uint4*>(
                                     k_base +
                                     static_cast<std::size_t>(key) * kv_width +
                                     d8)
                               : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  KRegs k_cur;
  VRegs v_cur;
  KRegs k_pre{};
  VRegs v_pre{};
  if (slot_begin < slot_end) {
    load_k(key_of(slot_begin), k_cur);
    load_v(key_of(slot_begin), v_cur);
  }

  for (std::uint32_t slot = slot_begin; slot < slot_end; slot += kKeys) {
    const std::uint32_t key_start = key_of(slot);
    __syncthreads();
    if constexpr (kQ8) {
      uint4 halves[2];
      kv_q8::Dequantize16(k_cur, halves);
      auto* dst = reinterpret_cast<uint4*>(&kv_lds[q8_key * kKStride + q8_dim]);
      dst[0] = halves[0];
      dst[1] = halves[1];
    } else {
#pragma unroll
      for (std::uint32_t n = 0; n < kKRegs; ++n) {
        const std::uint32_t idx = tid + n * 256;
        *reinterpret_cast<uint4*>(&kv_lds[(idx / (kHeadDim / 8)) * kKStride +
                                          (idx % (kHeadDim / 8)) * 8]) =
            k_cur[n];
      }
    }
    __syncthreads();

    if (slot + kKeys < slot_end) {
      load_k(key_of(slot + kKeys), k_pre);
      load_v(key_of(slot + kKeys), v_pre);
    }

    {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = (s_kh * kKStepsPerWave + ks) * 16;
        s_acc = Wmma(q_frag[ks], LoadFrag(&kv_lds[sub * kKStride + d0]), s_acc);
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i)
        s_lds[s_kh][s_rb][(2 * i) + half_id][sub] = s_acc[i];
    }
    __syncthreads();

    // Online softmax: four threads per row, each over four keys.
    {
      const std::uint32_t rg = tid / kSoftmaxLanes;
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const bool live = row_position(rg) < positions;
      const std::uint32_t limit = start_pos + row_position(rg);
      float part_max = -INFINITY;
      float vals[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = seg * kPerLane + m;
        const std::uint32_t key = key_start + col;
        const bool valid = live && key <= limit && key < key_end;
        float score = s_lds[0][rb][row][col];
#pragma unroll
        for (std::uint32_t h = 1; h < kDimSplits; ++h)
          score += s_lds[h][rb][row][col];
        vals[m] = valid ? score : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_max = fmaxf(part_max, __shfl_xor(part_max, off));
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p_lds[rg][seg * kPerLane + m] = static_cast<__half>(w);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1)
        part_sum += __shfl_xor(part_sum, off);
      running_max = next_max;
      running_sum = (running_sum * prior_scale) + part_sum;
      if (seg == 0)
        row_scale[rg] = prior_scale;
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const float scale = row_scale[(rb * 16) + (2 * i) + half_id];
        o_acc[rb][0][i] *= scale;
        o_acc[rb][1][i] *= scale;
      }
    }

    // V^T: lane L owns key L % 16 and each thread a 16-dimension slice, so
    // consecutive lanes write consecutive columns (conflict-free).
    {
      uint4 v_halves[kVRegs];
      if constexpr (kQ8) {
        kv_q8::Dequantize16(v_cur, v_halves);
      } else {
#pragma unroll
        for (std::uint32_t j = 0; j < kVRegs; ++j)
          v_halves[j] = v_cur[j];
      }
#pragma unroll
      for (std::uint32_t j = 0; j < kVRegs; ++j) {
        const auto* packed = reinterpret_cast<const __half*>(&v_halves[j]);
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          kv_lds[(v_slice + j * 8 + i) * kVtStride + v_key] = packed[i];
      }
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + t * 8;
      const v16h v_frag = LoadFrag(&kv_lds[(dim_tile * 16 + sub) * kVtStride]);
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
        const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][0]);
        v8f next = Wmma(p_frag, v_frag, o_acc[rb][t]);
        // Row block rb holds positions 2 rb and 2 rb + 1. A row whose own
        // position falls inside this tile sums only its visible keys, in
        // order, so it does not depend on what later keys hold (a single
        // decoded row has not written them yet). A row whose position
        // precedes the tile keeps its accumulator: its probabilities are all
        // zero there, and a single decoded row never visits the tile.
        if (key_start + kKeys - 1 > start_pos + 2 * rb) {
          v8f tail = o_acc[rb][t];
#pragma unroll
          for (std::uint32_t key = 0; key < kKeys; ++key) {
            const float value = static_cast<float>(v_frag[key]);
#pragma unroll
            for (std::uint32_t i = 0; i < 8; ++i) {
              const std::uint32_t row = rb * 16 + 2 * i + half_id;
              if (key_start + key <= start_pos + row_position(row))
                tail[i] = fmaf(__half2float(p_lds[row][key]), value, tail[i]);
            }
          }
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const std::uint32_t limit =
                start_pos + row_position(rb * 16 + 2 * i + half_id);
            if (key_start > limit)
              next[i] = o_acc[rb][t][i];
            else if (key_start + kKeys - 1 > limit)
              next[i] = tail[i];
          }
        }
        o_acc[rb][t] = next;
      }
    }

    if constexpr (kQ8) {
      k_cur = k_pre;
      v_cur = v_pre;
    } else {
#pragma unroll
      for (std::uint32_t n = 0; n < kKRegs; ++n)
        k_cur[n] = k_pre[n];
#pragma unroll
      for (std::uint32_t n = 0; n < kVRegs; ++n)
        v_cur[n] = v_pre[n];
    }
  }
  if (tid % kSoftmaxLanes == 0) {
    row_max[tid / kSoftmaxLanes] = running_max;
    row_sum[tid / kSoftmaxLanes] = running_sum;
  }
  __syncthreads();

  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row_scratch =
      static_cast<std::size_t>(num_heads) * max_splits * (kHeadDim + 2);
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim = (wave + t * 8) * 16 + sub;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = rb * 16 + 2 * i + half_id;
        if (row_position(row) >= positions)
          continue;
        float* base = scratch + row_position(row) * row_scratch;
        const std::size_t partition_index =
            static_cast<std::size_t>(row_head(row)) * max_splits + split;
        base[stats_elements + partition_index * kHeadDim + dim] =
            o_acc[rb][t][i];
      }
    }
  }
  if (tid < kRows && row_position(tid) < positions) {
    float* base = scratch + row_position(tid) * row_scratch;
    const std::size_t partition_index =
        static_cast<std::size_t>(row_head(tid)) * max_splits + split;
    base[partition_index * 2 + 0] = row_max[tid];
    base[partition_index * 2 + 1] = row_sum[tid];
  }
}


}  // namespace dec

}  // namespace

template<std::uint32_t kQueryHeads, std::uint32_t kKvHeads>
static bool LaunchSparseGeometry(
    const float* q, const float* gate, void* k_cache_f16, void* v_cache_f16,
    float* out_context, void* pooled_keys, std::int32_t* block_lists,
    std::uint32_t* pooled_valid_end, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    const SparseAttentionConfig& config, hipStream_t stream) {
  if (head_dim != kHeadDim || batch_size == 0 || pooled_keys == nullptr ||
      block_lists == nullptr || max_context > kSelectMaxBlocks * kSparseBlockKeys ||
      start_pos + batch_size > max_context) {
    return false;
  }
  constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
  constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;
  const std::uint32_t max_blocks =
      (max_context + kSparseBlockKeys - 1) / kSparseBlockKeys;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kKvWidth;
  const auto* layer_k = static_cast<const __half*>(k_cache_f16) + layer_cache_offset;
  const auto* layer_v = static_cast<const __half*>(v_cache_f16) + layer_cache_offset;
  auto* pooled = static_cast<__half*>(pooled_keys) +
                 static_cast<std::size_t>(layer_idx) * kKvHeads * max_blocks * 512;

  const std::uint32_t end_pos = start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t valid_end = *pooled_valid_end;
  const std::uint32_t from_block = min(valid_end, start_pos) / kSparseBlockKeys;
  const std::uint32_t to_block = (end_pos - 1) / kSparseBlockKeys;
  UpdatePooledKeys<<<dim3(to_block - from_block + 1, kKvHeads), kHeadDim, 0,
                     stream>>>(layer_k, pooled, from_block, end_pos, kKvHeads,
                               max_blocks);
  *pooled_valid_end = end_pos;

  const std::uint32_t tiles = static_cast<std::uint32_t>(
      (batch_size + kWmmaQueryRows - 1) / kWmmaQueryRows);
  SelectSparseBlocks<kQueryHeads, kKvHeads>
      <<<dim3(tiles, kKvHeads), 256, 0, stream>>>(
          q, pooled, block_lists, start_pos, batch_size, max_blocks,
          config.sink_blocks, config.local_blocks, config.top_blocks,
          SparseAttentionListStride(config));
  SparseWmmaCausalAttention<kWmmaQueryRows, kWmmaKeys, false, kQueryHeads,
                            kKvHeads>
      <<<dim3(tiles, kKvHeads * (kGqa / kWmmaHeads)), 256, 0, stream>>>(
          q, gate, layer_k, layer_v, out_context, start_pos, batch_size,
          block_lists, SparseAttentionListStride(config));
  return true;
}

bool LaunchQwenSparseWmmaAttention(
    const float* q, const float* gate, void* k_cache_f16, void* v_cache_f16,
    float* out_context, void* pooled_keys, std::int32_t* block_lists,
    std::uint32_t* pooled_valid_end, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t batch_size, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    const SparseAttentionConfig& config, hipStream_t stream) {
  if (num_heads == 24 && num_kv_heads == 4) {
    return LaunchSparseGeometry<24, 4>(
        q, gate, k_cache_f16, v_cache_f16, out_context, pooled_keys,
        block_lists, pooled_valid_end, layer_idx, start_pos, batch_size,
        max_context, num_heads, num_kv_heads, head_dim, config, stream);
  }
  if (num_heads == 16 && num_kv_heads == 2) {
    return LaunchSparseGeometry<16, 2>(
        q, gate, k_cache_f16, v_cache_f16, out_context, pooled_keys,
        block_lists, pooled_valid_end, layer_idx, start_pos, batch_size,
        max_context, num_heads, num_kv_heads, head_dim, config, stream);
  }
  return false;
}


bool LaunchQwenSparseDecodeAttention(
    const float* q, const float* gate, void* k_cache_f16, void* v_cache_f16,
    float* out_context, void* pooled_keys, std::int32_t* block_lists,
    std::uint32_t* pooled_valid_end, std::uint32_t layer_idx,
    std::uint32_t start_pos, std::size_t positions, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads, std::uint32_t head_dim,
    const SparseAttentionConfig& prefill_config, hipStream_t stream,
    std::span<float> scratch) {
  constexpr std::uint32_t kQ = 16;
  constexpr std::uint32_t kKv = 2;
  constexpr std::uint32_t kMaxSplits = detail::kSplitKDecodeAttentionMaxSplits;
  SparseAttentionConfig config = prefill_config;
  config.top_blocks = prefill_config.decode_top_blocks;
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(num_heads, head_dim);
  if (num_heads != kQ || num_kv_heads != kKv || head_dim != kHeadDim ||
      positions == 0 || positions > kWmmaQueryRows || pooled_keys == nullptr ||
      block_lists == nullptr || scratch.size() < positions * row_scratch ||
      max_context > kSelectMaxBlocks * kSparseBlockKeys ||
      start_pos + positions > max_context) {
    return false;
  }
  constexpr std::uint32_t kKvWidth = kKv * kHeadDim;
  const std::uint32_t max_blocks =
      (max_context + kSparseBlockKeys - 1) / kSparseBlockKeys;
  const std::size_t layer_cache_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kKvWidth;
  const auto* layer_k = static_cast<const __half*>(k_cache_f16) + layer_cache_offset;
  const auto* layer_v = static_cast<const __half*>(v_cache_f16) + layer_cache_offset;
  auto* pooled = static_cast<__half*>(pooled_keys) +
                 static_cast<std::size_t>(layer_idx) * kKv * max_blocks * 512;

  const std::uint32_t end_pos =
      start_pos + static_cast<std::uint32_t>(positions);
  const std::uint32_t valid_end = *pooled_valid_end;
  const std::uint32_t from_block = min(valid_end, start_pos) / kSparseBlockKeys;
  const std::uint32_t to_block = (end_pos - 1) / kSparseBlockKeys;
  UpdatePooledKeys<<<dim3(to_block - from_block + 1, kKv), kHeadDim, 0,
                     stream>>>(layer_k, pooled, from_block, end_pos, kKv,
                               max_blocks);
  *pooled_valid_end = end_pos;
  SelectSparseBlocks<kQ, kKv><<<dim3(1, kKv), 256, 0, stream>>>(
      q, pooled, block_lists, start_pos, positions, max_blocks,
      config.sink_blocks, config.local_blocks, config.top_blocks,
      SparseAttentionListStride(config));

  const std::size_t attn_width = static_cast<std::size_t>(kQ) * kHeadDim;
  for (std::size_t first = 0; first < positions; first += dec::kPositions) {
    const auto count = static_cast<std::uint32_t>(
        std::min<std::size_t>(dec::kPositions, positions - first));
    dec::SparseDecodePartials<false>
        <<<dim3(kMaxSplits, kKv), 256, 0, stream>>>(
            q + first * attn_width, layer_k, layer_v,
            scratch.data() + first * row_scratch,
            start_pos + static_cast<std::uint32_t>(first), count, block_lists,
            SparseAttentionListStride(config), kQ, kKv);
  }
  LaunchSplitKAttentionReduce(gate, scratch.data(), out_context, kQ,
                              static_cast<std::uint32_t>(positions), kMaxSplits,
                              stream);
  return true;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
