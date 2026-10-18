# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- SFT expansion to 20K pairs (pipeline ready, data-only)
- Full 20M-token pretrain (currently 10M)
- Keypoint heatmap to replace decode+argmax
- Learned policy network to replace MPC

## [0.1.0] — 2026-10-18

First public release. World model + language model + embodied agent closed loop,
all trained from scratch, reproducible on a single GPU.

### Added

**World model**
- Encoder CNN / Dynamics MLP / Decoder CNN triad (~1.5M params, 64×64 grayscale)
- v3 state head (z → position/velocity/target/distance) — better reconstruction (MSE 0.0018 vs 0.0133)
  but worse multi-step planning; shipped as `checkpoint_v3.pt` alongside the default v1
- MPC planning with `continuous` candidate set: exact aim + ±10°/±20° neighbors +
  16 continuous samples. Navigation **9/16 → 16/16**
- WorldState event system: reached / bumped / failed / timeout, quotable by the language layer

**Language models**
- `tinygpt.py`: GPT from scratch (~200 lines), char-level
- char 77.5M (chat + knowledge + world narration) — production default
- BPE 123.4M (16 layers / 768 dim) trained on a 30M-char industrial corpus
- Latent-state injection at `[STATE]` token: perception QA with zero hallucination,
  distance error 0.1-0.4
- 4-layer answer fallback: table lookup → retrieval (669 entries) → generation → "I don't know"

**Tokenizer**
- `bpe_tokenizer.py`: pure-Python BPE from scratch, incremental training
  (3M chars in 6.5 s, 100× faster than naive scan)
- `bpe_6400.json` shipped with the repo (industrial-corpus vocabulary)

**Agent**
- Single entry `agent.py` with CLI / HTTP API / MCP Server interfaces
- Parametric tool calls: `[调用工具 world_act, direction=西北, magnitude=0.7]`
- Remote HTTP tools via `register_http_tool()`
- Symbolic calculator (arithmetic never goes through the LM)

**Tooling**
- `eval_world.py` / `eval_planner.py` / `verify_state_bpe.py`: A/B harnesses (16 seeds)
- `bench_probe.py`: GPU throughput probe (found the ROCm large-batch pathology)
- `bench_probe` results: 123M @ 629 ms/step (batch 48, fp16); batch 128 is 22× slower
- 30 unit tests (CPU, ~4 s, no weights required)
- Pretrained weights on [Release v0.1.0](https://github.com/Alex663028/mini-world-model/releases/tag/v0.1.0)

### Known limitations
- One-step imagined frames are blurry (MSE mode averaging); v3 decoder is sharper but drifts faster
- 123M LM degrades into repetition in multi-turn agentic loops
- Parametric tool calls flip up/down; rule layer used in production
- No CI yet (requires a token with `workflow` scope)

### Notes
- Vocabulary is domain-dependent: 1.19 compression on general text, 2.22 on dialog-only corpora.
  Swapping vocabularies invalidates all BPE weights.
- ROCm: keep batch at 48; fp32 does not use matrix acceleration on RDNA2 (0.5 vs 18 TFLOPS).
