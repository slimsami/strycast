#include "src/families/qwen/hip_common/sparse_attention_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using gufo::hip::detail::kSparseQueryRows;
using gufo::hip::detail::PlanSparsePrefillChunk;
using gufo::hip::detail::SparseDecodeDenseRows;
using gufo::hip::detail::SparseDenseTailStart;

int failures{0};

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

// Mirrors ForwardPromptBatch: rows of a chunk below min_context run dense
// (the kernel's visible-context test), a chunk of at most eight rows takes the
// verification route, and the others share a selection per 32-row tile counted
// from the chunk start.
constexpr std::size_t kSmallChunkRows = 8;
constexpr std::size_t kMaxBatch = 4096;
constexpr std::size_t kMinContext = 32768;
constexpr std::size_t kTailRows = 1024;

struct RowRoute {
  enum Kind : std::uint8_t { kDense, kTail, kSparse, kSparseSmall } kind;
  std::size_t tile;  // first row of the shared selection (sparse rows)
  bool operator==(const RowRoute&) const = default;
};

// Routes every row of `[from, prompt_end)` for one serving pass grid: passes of
// `pass` tokens whose ends the pool rounds down to the alignment.
std::vector<RowRoute> Route(std::size_t from, std::size_t prompt_end,
                            std::size_t pass) {
  std::vector<RowRoute> routes(prompt_end);
  const std::size_t dense_from = SparseDenseTailStart(prompt_end, kTailRows);
  for (std::size_t offset = from; offset < prompt_end;) {
    std::size_t budget = std::min(pass, prompt_end - offset);
    if (const auto end = offset + budget; end < prompt_end &&
                                          end % kSparseQueryRows != 0 &&
                                          budget > end % kSparseQueryRows)
      budget -= end % kSparseQueryRows;
    const std::size_t call_end = offset + budget;
    for (std::size_t position = offset; position < call_end;) {
      const auto chunk = PlanSparsePrefillChunk(position, call_end, kMaxBatch,
                                                kMinContext, dense_from);
      const std::size_t chunk_end = position + chunk.rows;
      for (std::size_t row = position; row < chunk_end; ++row) {
        RowRoute route{RowRoute::kDense, 0};
        if (chunk.dense_tail) {
          route.kind = RowRoute::kTail;
        } else if (chunk_end > kMinContext) {
          route.kind = chunk.rows <= kSmallChunkRows ? RowRoute::kSparseSmall
                                                     : RowRoute::kSparse;
          route.tile = chunk.rows <= kSmallChunkRows
                           ? position
                           : position + ((row - position) / kSparseQueryRows) *
                                            kSparseQueryRows;
        }
        routes[row] = route;
      }
      position = chunk_end;
    }
    offset = call_end;
  }
  return routes;
}

void TestDenseTailStart() {
  Check(SparseDenseTailStart(70000, 1024) == 68960,
        "tail starts on the tile at or before the last 1024 rows");
  Check(70000 - SparseDenseTailStart(70000, 1024) >= 1024,
        "tail covers at least 1024 rows");
  Check(SparseDenseTailStart(500, 1024) == 0, "short prompt is all tail");
}

void TestChunksStopAtBoundaries() {
  const std::size_t dense_from = SparseDenseTailStart(70000, kTailRows);
  auto chunk =
      PlanSparsePrefillChunk(30720, 70000, kMaxBatch, kMinContext, dense_from);
  Check(chunk.rows == 2048 && !chunk.dense_tail, "chunk stops at min_context");
  chunk =
      PlanSparsePrefillChunk(66560, 70000, kMaxBatch, kMinContext, dense_from);
  Check(66560 + chunk.rows == dense_from, "chunk stops at the dense tail");
  chunk = PlanSparsePrefillChunk(dense_from, 70000, kMaxBatch, kMinContext,
                                 dense_from);
  Check(chunk.dense_tail && dense_from + chunk.rows == 70000,
        "tail chunk runs to the prompt end");
  chunk =
      PlanSparsePrefillChunk(40010, 70000, kMaxBatch, kMinContext, dense_from);
  Check(chunk.rows == 22, "unaligned sparse start stops at the next tile");
  chunk = PlanSparsePrefillChunk(40000, 70000, 1000, kMinContext, dense_from);
  Check(chunk.rows == 992, "batch-limited sparse chunk ends on a tile");
  chunk =
      PlanSparsePrefillChunk(40000, 41000, kMaxBatch, kMinContext, dense_from);
  Check(chunk.rows == 1000, "chunk reaching the call end keeps every row");
  chunk =
      PlanSparsePrefillChunk(1000, 5000, kMaxBatch, kMinContext, dense_from);
  Check(chunk.rows == 4000, "dense rows keep the plain batch split");
}

void TestRoutesIgnoreThePassGrid() {
  for (const std::size_t prompt_end : {40000U, 64460U, 77596U, 33000U}) {
    const auto reference = Route(0, prompt_end, kMaxBatch);
    for (const std::size_t pass : {512U, 1000U, 1024U, 2048U, 3072U, 4000U}) {
      Check(Route(0, prompt_end, pass) == reference,
            "pass size " + std::to_string(pass) + " routes prompt " +
                std::to_string(prompt_end) + " like one pass");
    }
    // A prefix restored from the cache sits on the alignment (the component
    // cache restores aligned checkpoints only): the rest routes the same.
    for (const std::size_t cached : {30720U, 32768U, 35840U}) {
      if (cached >= prompt_end)
        continue;
      const auto resumed = Route(cached, prompt_end, 2048);
      bool same = true;
      for (std::size_t row = cached; row < prompt_end; ++row)
        same = same && resumed[row] == reference[row];
      Check(same, "aligned cached prefix " + std::to_string(cached) +
                      " routes the rest like one pass");
    }
  }
}

void TestUnalignedFrontierRealigns() {
  // A reused frontier after decoding is unaligned: only its first partial
  // tile differs, every row from the next tile on routes as in one pass.
  const std::size_t prompt_end = 64460;
  const auto reference = Route(0, prompt_end, kMaxBatch);
  const std::size_t frontier = 40010;
  const auto resumed = Route(frontier, prompt_end, 2048);
  bool same = true;
  for (std::size_t row = 40032; row < prompt_end; ++row)
    same = same && resumed[row] == reference[row];
  Check(same, "rows after the first tile route like one pass");
}

void TestDecodeDenseRows() {
  Check(SparseDecodeDenseRows(32760, 8, 32768) == 8,
        "segment ending at the threshold stays dense");
  Check(SparseDecodeDenseRows(32765, 8, 32768) == 3,
        "segment crossing the threshold splits there");
  Check(SparseDecodeDenseRows(32768, 8, 32768) == 0,
        "segment from the threshold is sparse");
}

}  // namespace

int main() {
  TestDenseTailStart();
  TestChunksStopAtBoundaries();
  TestRoutesIgnoreThePassGrid();
  TestUnalignedFrontierRealigns();
  TestDecodeDenseRows();
  if (failures != 0) {
    std::cerr << failures << " sparse attention plan test(s) failed.\n";
    return 1;
  }
  std::cout << "Sparse attention plan tests passed.\n";
  return 0;
}
