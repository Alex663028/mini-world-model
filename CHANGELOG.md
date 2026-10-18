# Changelog

All notable changes to this project. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- SFT expansion to 20K pairs (pipeline ready, data-only)
- Full 20M-token pretrain (currently 10M)
- Keypoint heatmap to replace decode+argmax
- Learned policy network to replace MPC

### Fixed
- **P0 — tokenizer OOV pollution**: `BPE.encode` no longer maps out-of-vocabulary characters to
  id 0 (id 0 is `"\n"`), which silently injected fake newlines into training data
  (`encode('你好😀世界𠮷')` used to decode back to `"你好\n世界\n"`). OOV characters are now
  dropped and counted (`oov_total` / `oov_chars` / `oov_rate`); `strict=True` raises `OOVError`.
  Added `python bpe_tokenizer.py --check --corpus <file>` to audit the drop rate before training,
  and `tokenize_corpus.py` reports it after encoding.
- **P0 — dead parameter `aim_horizon`**: `eval_planner.py --horizons` wrote `aim_horizon`, which
  nothing in the repo ever read, so every "horizon" run silently re-measured horizon=12 and the
  README's "imagination window is saturated" claim had no valid experiment behind it. It now
  writes `imagine_horizon`; the parameter is removed from `CONFIG` and the claim was re-measured
  (16 seeds: horizon 20 → 16/16, 26.3 steps, 55s; horizon 12 → 16/16, 26.3 steps, 33s).
- **P1 — empty corpus produced an empty vocabulary silently**: `bpe_tokenizer.py --train` now
  exits non-zero and writes nothing when the corpus is missing or shorter than 200 chars;
  `train_bpe` / `train_bpe_fast` raise `ValueError` on empty text.
- **P1 — wall-hit events were never recorded**: `agent._step_env` re-derived the physics formula
  and used a deviation threshold (>0.25) that normal action gains can never reach (~0.08 max).
  `PointMassEnv.step` now returns `info["hit_wall"]` and the agent consumes it; the duplicated
  formula is gone from the agent layer.
- **P1 — MCP robustness**: the 9 tools catch their own errors and return readable messages
  (`{"ok": false, "error": ...}` for the dict-returning tool); a missing `mcp` package now prints
  an install hint instead of an ImportError traceback.
- **P1 — narrowed exception handling**: `agent._load_lm` only swallows expected load errors and
  logs a full traceback; `reply()` keeps its boundary catch-all but logs full tracebacks, so a
  degraded answer no longer hides the root cause.
- **P2 — HTTP API**: malformed JSON now returns 400 (was a silent empty body + 200), oversized
  bodies 413, optional `--api-key` / `MINIWORLD_API_KEY` authentication (401), 500 responses log
  a traceback, and the server is a `ThreadingHTTPServer` guarded by a global lock (requests
  serialize on the single stateful env; the default bind stays `127.0.0.1`).
- **P2 — deprecated dead code**: `chat.ChatEngine` is unused (superseded by
  `agent.MiniWorldAgent`) and now emits a `DeprecationWarning`.
- **P3 — `torch.load`**: all 16 call sites now pass `weights_only=True` explicitly (verified
  against every released checkpoint).
- **P3 — downloads**: `download_corpus.py` falls back to `urllib` when `curl` is unavailable and
  fails loudly on empty output files.
- **P3 — `verify_state_bpe.py`**: missing `checkpoint.pt` produces a clear message, and the
  hit-rate denominator follows `--seeds` instead of a hard-coded `/4`.
- **P3 — single source for physics constants**: `DAMPING` / `ACTION_GAIN` / `DT` /
  `WALL_RESTITUTION` / `REACH_DIST` live in `env.py`; the literals copied into the agent and
  training code are gone.
- **P3 — packaging/docs**: added `pyproject.toml` (flat-layout `py-modules` list + pytest config);
  `pyarrow` and `mcp` are declared in `requirements.txt`; README updated for the API security
  options, the OOV policy and the corrected horizon numbers.

### Changed
- Unit tests: 30 → 41 (OOV behaviour, empty-corpus rejection, wall/arrival events, system table
  lookup, knowledge retrieval, dead-parameter regression). Windows-friendly subprocess decoding.

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
