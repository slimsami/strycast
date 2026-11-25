#ifndef GUFO_FAMILIES_QWEN_HIP_COMMON_SPARSE_ATTENTION_PLAN_HPP_
#define GUFO_FAMILIES_QWEN_HIP_COMMON_SPARSE_ATTENTION_PLAN_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gufo::hip::detail {

/// Block-sparse attention (STRYCAST_SPARSE_ATTN) routes every row by its
/// absolute position, never by where the chunk holding it starts, so a prompt
/// gives the same output whatever the prefill grid (pass size, checkpoint
/// captures, a reused prefix):
/// - rows below `min_context` and from the dense tail on take the dense
///   kernels, the rows between them the sparse ones;
/// - prefill queries share a selection in tiles of kSparseQueryRows counted
///   from position 0, so sparse chunks end on a tile and the serving runner
///   keeps pass ends and checkpoints there (prefill_alignment).
inline constexpr std::uint32_t kSparseQueryRows{32};

/// No dense tail (the prompt end is unknown and no logits are wanted).
inline constexpr std::size_t kNoSparseDenseTail{
    std::numeric_limits<std::size_t>::max()};

/// First position of the dense tail of a prompt ending at `prompt_end`: at
/// least its last `tail_rows` rows, from the tile boundary at or before them.
[[nodiscard]] constexpr std::size_t SparseDenseTailStart(
    std::size_t prompt_end, std::size_t tail_rows) noexcept {
  const std::size_t from = prompt_end > tail_rows ? prompt_end - tail_rows : 0;
  return from - (from % kSparseQueryRows);
}

struct SparsePrefillChunk {
  std::size_t rows{0};
  /// The rows belong to the dense tail (run with the sparse route suspended).
  bool dense_tail{false};
};

/// The next prefill chunk from absolute `position` (before `end`, at most
/// `max_rows`): it never crosses `min_context` or `dense_from`, and inside the
/// sparse range its tiles sit on multiples of kSparseQueryRows: it ends on a
/// tile unless it reaches the range end, and one starting off a tile ends at
/// the next one.
[[nodiscard]] constexpr SparsePrefillChunk PlanSparsePrefillChunk(
    std::size_t position, std::size_t end, std::size_t max_rows,
    std::size_t min_context, std::size_t dense_from) noexcept {
  const bool dense_tail = position >= dense_from;
  std::size_t limit = end;
  if (!dense_tail && dense_from < limit)
    limit = dense_from;
  if (position < min_context && min_context < limit)
    limit = min_context;
  std::size_t rows = std::min(max_rows, limit - position);
  if (!dense_tail && position >= min_context) {
    // A chunk starting off a tile (a reused frontier) stops at the next one.
    if (const std::size_t into = position % kSparseQueryRows; into != 0)
      rows = std::min(rows, kSparseQueryRows - into);
    if (position + rows < limit) {
      const std::size_t over = (position + rows) % kSparseQueryRows;
      if (rows > over)
        rows -= over;
    }
  }
  return {.rows = rows, .dense_tail = dense_tail};
}

/// Leading rows of a decode or verification segment of `rows` rows from
/// `start` that stay dense: a row at position p takes the sparse route only
/// once p >= min_context, as a single decode step does.
[[nodiscard]] constexpr std::size_t SparseDecodeDenseRows(
    std::size_t start, std::size_t rows, std::size_t min_context) noexcept {
  return start >= min_context ? 0 : std::min(rows, min_context - start);
}

}  // namespace gufo::hip::detail

#endif  // GUFO_FAMILIES_QWEN_HIP_COMMON_SPARSE_ATTENTION_PLAN_HPP_
