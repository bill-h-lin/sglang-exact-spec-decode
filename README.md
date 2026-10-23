# Bit-exact speculative decoding: Qwen3.8-27B on SGLang

I compare quantizations and inference engines on task quality. That only works if the engine
produces the same output when I haven't changed anything. If the engine drifts on its own, I can't
tell that apart from the effect I'm trying to measure.

This repo is the evidence and the fix for one case of that drift: speculative decoding on a hybrid
GDN model, where "lossless" is usually claimed without checking the token ids.

Measured on one RTX PRO 6000 Blackwell Max-Q (SM120), Qwen3.8-27B, BF16 weights and BF16 KV,
temperature 0, one request at a time unless noted.

## The question

Speculative decoding is supposed to return exactly what the target model would have returned on its
own. Does it, bit for bit? And if it doesn't, can you even detect it, or does it hide inside
"close enough" text that still reads fine?

## What I found

First, a control that surprised me. Before any of my patches, on vLLM nightly (DFlash2 k=9 + XQA):

- speculation on vs off: 15 of 24 prompts give different token ids
- `--max-num-seqs` 1 vs 32, no speculation at all: 21 of 24 differ
- a different attention backend: 20 of 24 differ

So a capacity setting moved more outputs than speculation did. The divergences all sat where the
top-1 and top-2 logits were within 0.25 of each other (0.125 for the backend and `--max-num-seqs`
changes), which is where rounding order decides the token. I recorded the `--max-num-seqs`
mechanism as unexplained rather than asserting a cause, because I have not isolated it.

On SGLang, turning on deterministic mode alone was not enough. With speculation on, 19 of 24 prompts
still gave different token ids and all 24 had different logprobs from no-speculation. Two changes to
speculative verify fix it (patch 1):

1. Verify runs the GDN recurrence through the decode kernel. The old verify kernel tiled the state
   in a different order and kept beta in fp32, where decode rounds it to bf16.
2. Verify attention runs as decode rows, one per draft position.

With both, the output matches no-speculation exactly: every token id and every logprob, on all 24
prompts, with every draft forced to reject (24/24), with 8 requests in flight (24/24), and on three
long prompts of 2,183, 31,975 and 127,735 tokens (3/3).

Exact run-to-run repeatability needs one more fix (patch 3). DeepGEMM's sm120 bf16 GEMM can release a
pipeline stage before the warp's reads of it land, so a run can differ from the next one: 25 of 1,200
request-runs with 8 in flight, and single prompts of 560 to 740 tokens in 5 to 15% of runs. With the
fence: 0 of 1,200 and 0 of 300.

## The cost

Deterministic mode is slow on its own (18.0 t/s against 26.4 stock, no speculation) because
DeepGEMM's sm120 bf16 kernel reads the weights at about a third of DRAM bandwidth at small batch.
Patch 2 routes those small-M GEMMs to a Triton tile with the same bits, which is also upstream as
[sgl-project/sglang#42145](https://github.com/sgl-project/sglang/pull/42145).

Decode t/s, single stream, temp 0, one RTX PRO 6000 Blackwell Max-Q:

| | short prompts | 2.2k tokens | 128k tokens |
|---|---:|---:|---:|
| SGLang, no speculation | 26.4 | 26.6 | 23.2 |
| vLLM, DFlash2 k=9 + XQA (not exact) | 109.8 | 67.4 | 56.1 |
| SGLang deterministic + DFlash 8 + these patches + #42145 (exact) | 106.6 | 73.9 | 42.6 |

Exact is 3% behind vLLM on short prompts, ahead at 2k, 24% behind at 128k. Prefill is slower too:
68.5 s to first token on a 128k prompt against 44.3 s on vLLM.

## Reproduction limits

Not tested: other models, other GPUs, tree verify, more than 8 requests in flight, temperature above
0. The numbers are single-stream at temp 0 on one card. The vLLM runs had a LoRA loaded but never
queried.

## The patches

See [`patches/`](patches/). Base: SGLang `main` at `542a043` (2026-09-23), DeepGEMM 0.2.0.

1. `1-sglang-exact-verify.patch` -- the two verify changes above. On with
   `--enable-deterministic-inference`; the default path does not change.
2. `2-sglang-small-m-route.patch` -- speed only, the small-M Triton route. On with
   `SGLANG_BATCH_INVARIANT_OPS_SMALL_M_TRITON=16`. Upstream (rebased since; this file is the version
   I measured): sgl-project/sglang#42145.
3. `3-deepgemm-sm120-fence.patch` -- the pipeline fence. Upstream fix (the fence alone) is
   [sgl-project/DeepGEMM#93](https://github.com/sgl-project/DeepGEMM/pull/93) by Ting SUN, merged into
   `dev` 2026-10-03; released in 0.2.3 (2026-10-16), which SGLang `main` has pinned since 2026-10-18.
   On the recipe's base (DeepGEMM 0.2.0) keep this patch; on 0.2.3 or later skip it: it won't apply.
   This file also adds `__syncwarp()`, and that is what I measured.

### Apply

```
P=/path/to/sglang-exact-spec-decode/patches
cd sglang && git checkout 542a043
git apply "$P/1-sglang-exact-verify.patch" "$P/2-sglang-small-m-route.patch"
pip install -e "python"

SP=$(python -c 'import deep_gemm, os; print(os.path.dirname(os.path.dirname(deep_gemm.__file__)))')
patch -p1 -d "$SP" < "$P/3-deepgemm-sm120-fence.patch"
```

A pip reinstall or upgrade of `sgl-deep-gemm` drops patch 3. DeepGEMM compiles kernels at run time, so
point `SGLANG_DG_CACHE_DIR` at an empty directory the first time, or the old kernel can come back from
the cache.

### Launch

Model `Qwen/Qwen3.8-27B` at `1d4bf0f2`, BF16. Draft `incoai/Qwen3.8-27B-DFlash2` at `015e7956`.

```
SGLANG_BATCH_INVARIANT_OPS_SMALL_M_TRITON=16 SGLANG_DG_CACHE_DIR=/path/to/empty/dir \
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
  --enable-deterministic-inference --attention-backend triton \
  --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8 --mamba-ssm-dtype float32 \
  --max-running-requests 8 --chunked-prefill-size 8192 --mem-fraction-static 0.88 \
  --context-length 262144 --disable-radix-cache
```

## Write-up

The r/LocalLLaMA post this accompanies:
https://www.reddit.com/r/LocalLLaMA/comments/1x3sx3d/

## License

Apache-2.0, matching SGLang. The patches are derivative of SGLang (Apache-2.0) and DeepGEMM (MIT).
