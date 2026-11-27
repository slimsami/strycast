# Qwen3.8 27B implementation notes

Developer detail behind the [public model page](../../../models/qwen3.8-27b.md).
[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)
· [ROCm 10.1 migration](ROCM.md)

BF16 targets are reference-only. Targets and drafts need not have matching
precision; the Q4_K_M DFlash2 draft serves both targets.

## Wide copy verification

A request decoding alone can verify a prompt-lookup copy wider than the DFlash2
block: greedy requests, and sampled ones without a seed. Copy-only steps start
at the ordinary eight rows; two whole copies in a row at the current width
double it, up to 64 rows, and any rejected copy returns it to eight. The wide
pass shares every projection across its rows and runs attention in eight-row
segments, so each row's logits and DFlash2 target features equal those of
consecutive ordinary verifications (`qwen27b_wide_verification_test` also checks
decoding after cutting back to any kept prefix). Rollback replays the captured
recurrent inputs of every kept row, so the session's SSM replay ring grows from
16 to 64 rows (about 91 MiB) on first use; the resource claim includes it, and
the load narrows the widest pass when the configured sessions would not fit.
When more than eight rows are kept, the draft injects all but the last at once,
so snapshots and batched steps still hold at most eight pending rows. Seeded
sampled requests and batched steps keep the ordinary width.

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](QUALITY.md).

## 8-bit KV cache

`gufo serve llm --kv-cache q8_0` works for the 27B as for the 35B-A3B
([format and design](../qwen3.6-35b-a3b/README.md#8-bit-kv-cache)): the same
writers and the same prefill kernels, which already took any head grouping.
Only the matrix-core decode/verify kernel was written for eight query heads
per KV head. For the 27B's six it runs five positions per block, 30 of its 32
rows, the last two being padding no position owns. FP16 decoding of the 27B
does not take that kernel and is unchanged.

The 27B's 16 attention layers hold 4 KV heads each, so Q8 saves 3.75 GiB per
session at context 131072 (7.5 GiB at 262144); measured on a 64K session,
23,668 → 21,748 MiB of device memory. CUDA-graph decode is off in Q8 mode, as
for the 35B; decode reads half the KV bytes instead.

Measured on a Radeon 890M (gfx1150), UD-Q4_K_XL, one session at 65536 (Nov
27, 2026):

| Check | f16 | q8_0 |
| --- | --- | --- |
| Greedy DFlash2 vs AR, 8 prompts | 8/8 identical | 8/8 identical |
| Greedy AR answers vs f16 | | 7/8 identical (one differs from the 3rd character) |
| Four needles in a 31.5K-token prompt | 4/4 | 4/4 |
| AR decode, short prompts, cool GPU | 4.4 tok/s | 4.4 tok/s |
| DFlash2 decode, mean of 6 prompts | 11.4 tok/s | 11.7 tok/s |
| Prefill, 3.5K / 15K tokens | 161 / 163 tok/s | 154 / 163 tok/s |

AR decode on that machine falls to about 3 tok/s in either format after a few
minutes of sustained load (f16 run after q8_0 showed the same fall), so
compare AR speeds from runs that start cool. `qwen_attention_kv_q8_ops_test`
covers both head shapes, including blocks of five positions and verification
rows against single decoding.

## Vision

Vision weights upload lazily. In `chat`, `--image` attaches to the first user
turn. Ordered text/image parts, multiple images and later turns work with
AR/DFlash2 and caches. Official dynamic resizing allows 64–16384 merged image
tokens. Cache identity includes processed pixels, placement, preprocessing
version and projector contents. See [vision checks and limits](QUALITY.md#vision).
