#if defined(ENGINE_ENABLE_HIP)
#include <algorithm>
#include <chrono>
#include <gufo/ext/experimental/hip_utils.hpp>
#include <stdexcept>
#include <thread>

#include "src/models/qwen/hip/executor.hpp"

namespace gufo::hip {
void QwenGpuExecutor::CheckPrefillCancellation() const {
  if (cancellation_check_ && cancellation_check_())
    throw QwenPrefillCancelled();
}

void QwenGpuExecutor::SetPrefillYield(std::size_t groups,
                                      std::function<void()> yield) {
  yield_layers_.clear();
  yield_ = {};
  if (groups < 2 || !yield)
    return;
  const std::uint32_t layers = weights_.config.num_layers;
  for (std::size_t k = 1; k < groups; ++k) {
    const auto layer = static_cast<std::uint32_t>(k * layers / groups);
    if (layer != 0 && layer < layers &&
        (yield_layers_.empty() || yield_layers_.back() != layer))
      yield_layers_.push_back(layer);
  }
  yield_ = std::move(yield);
}

void QwenGpuExecutor::PrefillLayerCheckpoint(std::uint32_t layer) {
  // Every buffer this pass carries across layers is in this executor's
  // arena, which no other session's decode touches. Draining the queued
  // layers first keeps them from sharing the GPU with the decode round.
  if (yield_ && std::ranges::binary_search(yield_layers_, layer)) {
    HIP_CHECK(hipStreamSynchronize(arena_.stream));
    yield_();
  }
  CheckPrefillCancellation();
  // Keep one four-layer group ahead of the GPU. Without backpressure, AR
  // can enqueue the whole prompt before the client disconnects; checking
  // cancellation on the host then cannot stop seconds of abandoned work.
  if (!cancellation_check_ || layer == 0 || layer % 4 != 0)
    return;
  for (auto& event : prefill_events_) {
    if (!event)
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  }
  const auto slot = (layer / 4 - 1) % prefill_events_.size();
  HIP_CHECK(hipEventRecord(prefill_events_[slot], arena_.stream));
  if (layer < 8)
    return;
  hipError_t status;
  while ((status = hipEventQuery(prefill_events_[1 - slot])) ==
         hipErrorNotReady) {
    CheckPrefillCancellation();
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  HIP_CHECK(status);
}

tokenization::TokenId QwenGpuExecutor::ForwardSmallPromptChunk(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  // A few rows cost about as much as one prefill pass over every weight; the
  // verification forward reads each weight once per row group instead.
  const auto predictions =
      ForwardDecodeEquivalentVerificationChunk(prompt_tokens, start_pos, false);
  if (capture_prompt_hidden_) {
    const std::size_t old_size = h_prompt_hidden_.Size();
    h_prompt_hidden_.Resize(old_size + h_verification_hidden_.size());
    std::copy(h_verification_hidden_.begin(), h_verification_hidden_.end(),
              h_prompt_hidden_.Data() + old_size);
  }
  h_verification_hidden_.clear();
  last_verification_rows_ = 0;
  if (!compute_logits)
    return 0;
  const std::size_t vocab_size = weights_.config.vocab_size;
  auto scratch = arena_.GetScratchView();
  HIP_CHECK(hipMemcpyAsync(
      scratch.decode.logits.data(),
      d_verification_logits_ + (prompt_tokens.size() - 1) * vocab_size,
      vocab_size * sizeof(float), hipMemcpyDeviceToDevice, arena_.stream));
  // The verification forward commits without logits; the prompt frontier has
  // them now, so a checkpoint captured here can decode its next token.
  if (continuation_hooks_)
    continuation_hooks_->AfterLogits();
  return predictions.back();
}

tokenization::TokenId QwenGpuExecutor::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t start_pos, bool compute_logits) {
  CheckReset();
  CheckPrefillCancellation();
  if (prompt_tokens.empty()) {
    return 0;
  }
  replaying_ssm_state_ = false;
  arena_.DisableSsmReplayCapture();
  if (capture_prompt_hidden_) {
    h_prompt_hidden_.Clear();
    const std::size_t captured_layers =
        std::max<std::size_t>(arena_.GetTargetLayerCapture().size(), 1U);
    const std::size_t row = captured_layers * weights_.config.hidden_size;
    // Pinned up to one prompt batch (the serve path's largest call); a longer
    // prompt from a non-serve caller captures into pageable memory instead.
    h_prompt_hidden_.Reserve(prompt_tokens.size() * row,
                             std::size_t{GetMaxPromptBatch()} * row);
  }

  const std::size_t end_pos =
      static_cast<std::size_t>(start_pos) + prompt_tokens.size();
  if (end_pos > arena_.GetMaxContext()) {
    throw std::length_error("prompt exceeds the GPU context length");
  }

  tokenization::TokenId next_token = 0;
  // With block-sparse attention the last rows of a prompt (the question) are
  // run apart and dense: they have to find things anywhere in the context, and
  // a selection made from a tile's mean query does not reliably find them.
  // The tail is counted from the end of the whole prompt (set by the serving
  // runner, which prefills it in several calls), or from this call's end when
  // it computes the logits. Every row's route depends on its position only.
  std::size_t dense_from = detail::kNoSparseDenseTail;
  if (arena_.sparse_enabled) {
    const std::size_t prompt_end = arena_.sparse_prompt_end >= end_pos
                                       ? arena_.sparse_prompt_end
                                   : compute_logits ? end_pos
                                                    : 0;
    if (prompt_end != 0)
      dense_from =
          detail::SparseDenseTailStart(prompt_end, arena_.dense_tail_rows);
  }
  for (std::size_t offset = 0; offset < prompt_tokens.size();) {
    CheckPrefillCancellation();
    const auto plan =
        arena_.sparse_enabled
            ? detail::PlanSparsePrefillChunk(
                  start_pos + offset, end_pos, arena_.GetMaxBatch(),
                  arena_.sparse_config.min_context, dense_from)
            : detail::SparsePrefillChunk{
                  .rows = std::min<std::size_t>(arena_.GetMaxBatch(),
                                                prompt_tokens.size() - offset)};
    const bool tail_chunk = plan.dense_tail;
    const std::size_t chunk_size = plan.rows;
    const bool is_last = offset + chunk_size == prompt_tokens.size();
    struct Suspend {
      bool& flag;
      bool previous;
      Suspend(bool& f, bool on) : flag(f), previous(f) { flag = flag || on; }
      ~Suspend() { flag = previous; }
    } suspend(arena_.sparse_suspended, tail_chunk);
    const auto chunk = prompt_tokens.subspan(offset, chunk_size);
    const auto chunk_pos = start_pos + static_cast<std::uint32_t>(offset);
    next_token =
        chunk_size <= kSmallPromptChunkRows && weights_.config.IsMoE()
            ? ForwardSmallPromptChunk(chunk, chunk_pos,
                                      compute_logits && is_last)
            : ForwardPromptChunk(chunk, chunk_pos, compute_logits && is_last);
    offset += chunk_size;
  }
  return next_token;
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
