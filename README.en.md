# mini-world-model

[![Release](https://img.shields.io/github/v/release/Alex663028/mini-world-model)](https://github.com/Alex663028/mini-world-model/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](tests/)

**World model × language model × embodied agent — a complete closed loop, reproducible on a single GPU.**

A 6.7K-line, pure-PyTorch minimal embodied AI system: pixel perception → latent dynamics → MPC planning → action execution → event memory → language interaction.
Every model is **trained from scratch** (no `transformers` import), reproducible on a single RX 6800 / RTX 3090, with all measurements published — **including the failed experiments**.

```
Navigation success   9/16 → 16/16      World model recon   MSE 0.0018 (v3) / 0.0133 (v1)
BPE compression      1.19 (general) / 2.22 (dialog-only)
Perception QA        zero hallucination, distance error 0.1-0.4
Language models      char 77.5M → BPE 123.4M     Training speed  123M @ 629 ms/step (RX 6800)
```

[中文文档](README.md) · [Technical log (Chinese)](docs/ARCHIVE.md)

> Don't want to train? [Download pretrained weights](https://github.com/Alex663028/mini-world-model/releases/tag/v0.1.0) (790 MB), drop them in the repo root, and chat.

---

## What this is / isn't

**Is**: a teaching-grade implementation that connects all three layers — world model, language model, agent.
No wrappers, no API calls, readable code, reproducible data, measured claims.

**Isn't**: a product, a SOTA model, or anything that handles real images (input is 64×64 grayscale).
The language model cannot compete with any open-source small LLM. The value is in **seeing the whole system laid out**.

Useful if you:

- want to understand what a world model actually predicts (not just look at paper figures)
- want to build an agent from scratch without copying LangChain
- want to know when small models actually work and whether "add more parameters" helps
- are looking for an **honest** experiment log (most projects only report wins; this one reports losses too)

---

## Architecture

```
PointMassEnv  (64×64 grayscale, 2-D acceleration actions)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ① World model (~1.5M params, CNN + MLP)                │
│     obs ──Encoder CNN──> z_t (32-D latent)              │
│     action a_t ───────> Dynamics MLP ──> ẑ_{t+1}        │
│              ┌───────────┴───────────┐                  │
│              ▼                       ▼                  │
│        Decoder CNN              State Head (v3 exp.)    │
│        imagined frame           z → pos/vel/dist        │
└─────────────────────────────────────────────────────────┘
        │ z_t
        ▼
┌─────────────────────────────────────────────────────────┐
│  ② Planning & control                                   │
│     MPC (12-step imagination) ──> action ──> PD control  │
│     Candidates: exact aim + ±10°/±20° neighbors + 16 samples │
└─────────────────────────────────────────────────────────┘
        │ execution + events
        ▼
┌─────────────────────────────────────────────────────────┐
│  ③ Agent layer (agent.py, single entry)                 │
│     WorldState: pos/vel/target/distance + event stream   │
│     Events: reached / bumped / failed / timeout         │
│     Tools: world_perceive / world_navigate / world_act   │
│            world_imagine / kb_query / calc              │
│     Remote tools: register_http_tool() → any HTTP endpoint │
└─────────────────────────────────────────────────────────┘
        │ z_t injected at [STATE] token
        ▼
┌─────────────────────────────────────────────────────────┐
│  ④ Language model (char 77.5M / BPE 123.4M, from scratch) │
│     Perception QA: latent injection → "target is north, 0.7 away" │
│     Chat/knowledge: 4-layer fallback (table → retrieve → generate → say "I don't know") │
│     Tool calls: [调用工具 world_act, direction=西北]      │
└─────────────────────────────────────────────────────────┘
```

Three interfaces: **CLI / HTTP API / MCP Server**.

---

## Key results

| Capability | Metric | Notes |
|---|---|---|
| **Navigation success** | **16/16 (100%)** | 16 seeds, continuous candidates; fixed-8 baseline 9/16 |
| Navigation cost | 26.3 steps / 40 s | Average, including planning |
| World model recon. | MSE **0.0018** (v3) / 0.0133 (v1) | Foreground-weighted BCE, 64×64 |
| Open-loop imagination | error < 0.1 within 8 steps | diverges to ~0.8 at 30 steps (inherent to world models) |
| **Perception QA** | **zero hallucination**, distance error 0.1-0.4 | Latent injection, BPE 123M |
| Chat / knowledge | 11/11 normal | 4-layer fallback, 669-entry knowledge base |
| Tool calling | 3/5 autonomous decisions correct | "23×7 → calc", "长城 → kb_query" |
| BPE compression | **1.19 (general) / 2.22 (dialog)** | Domain-dependent vocabulary |
| BPE training speed | 3M chars in **6.5 s** | Incremental algorithm (naive scan: 10+ min) |
| Training speed | 123M @ **629 ms/step** | RX 6800, fp16, batch 48, block 256 |

---

## Quick start

### Route A — download pretrained weights (fastest)

Grab the files from [Release v0.1.0](https://github.com/Alex663028/mini-world-model/releases/tag/v0.1.0) into the repo root:

| File | Size | Purpose |
|---|---|---|
| `checkpoint.pt` | 6.4 MB | World model v1 (navigation 16/16) |
| `checkpoint_v3.pt` | 6.5 MB | World model v3 (sharper reconstruction) |
| `lm_unified.pt` | 289 MB | LM: chat + knowledge + world narration |
| `lm_state_bpe.pt` | 488 MB | Fused: BPE 123M + latent injection (perception) |

```bash
pip install -r requirements.txt
python agent.py "你好"                    # chat (lm_unified)
python agent.py "你看到了什么？"           # perception (auto-selects lm_state_bpe)
python agent.py "去目标那儿"               # navigation (world model + MPC)
python agent.py "23 乘以 7"                # symbolic calculator
```

Weights are auto-detected by name: when `lm_state.pt` is absent, the loader upgrades to
`lm_state_bpe.pt` (the fused model). The `bpe_6400.json` vocabulary ships with the repo.

### Route B — train from scratch (to learn each step)

```bash
pip install -r requirements.txt
python train.py --episodes 400 --epochs 20   # world model, ~4 min on CPU
python demo.py                               # 1-step prediction / open-loop rollout / MPC
```

**HTTP API** (the `/` endpoint self-documents every route):

```bash
python api_server.py --port 8399 &
curl http://127.0.0.1:8399/                                    # list all endpoints
curl -X POST http://127.0.0.1:8399/reply -d '{"text":"你好"}'
curl -X POST http://127.0.0.1:8399/task  -d '{"task":"去目标那儿"}'
curl http://127.0.0.1:8399/state                               # position / events / steps
```

**Tests** (30 cases, ~4 s on CPU, no weights needed):

```bash
pip install pytest
python -m pytest tests/ -q
```

**A/B harness** (run this before trusting any single-seed result):

```bash
python eval_world.py   --seeds 16                  # world model comparison
python eval_planner.py --seeds 16 --horizons 20    # planning strategy comparison
python verify_state_bpe.py --ckpt lm_state_bpe.pt  # perception QA comparison
python bench_probe.py                              # GPU throughput probe
```

---

## Modules

### ① World model: the classic triad, three generations

| Component | Here | DreamerV3 scale |
|---|---|---|
| Observation | 64×64 grayscale | 720p video |
| Encoder | 4-layer CNN | ViT / 3D VAE |
| Dynamics | 128-hidden MLP | hundreds of millions of Transformer params |
| Training data | 24K rollout frames | millions of hours of video |

Three experiments show it genuinely learned the dynamics: one-step prediction aligns agent
position and shape; open-loop rollout tracks for 8 steps before diverging; MPC reaches the goal.

**Two unsuccessful-but-informative attempts:**

| Version | Change | Outcome |
|---|---|---|
| v2 | 50% random + 50% MPC data, double data | 9/16 → 10/16 (+6%, noise level) → **data is not the bottleneck** |
| v3 | Added state head (z → pos/vel/target/dist) | Reconstruction MSE **0.0018** (7× better than v1), but MPC via state head is 10× less accurate |

**v3 in full** (8 seeds each):

| Config | Navigation | Note |
|---|---|---|
| v1 + decode+argmax | **8/8** | current default |
| v3 + state head | 5/8 | state-head RMSE 0.178 vs pixel 1/64 ≈ 0.016 |
| v3 + decode+argmax | 7/8 | better single-step recon, worse multi-step drift |

**Conclusion: best single-step reconstruction ≠ best multi-step planning.** v3's decoder is clearly
better (7× MSE) but its dynamics diverge faster over the 12-step imagination window, so MPC gets worse.
Choosing v1 as default is an engineering trade-off, not a claim of technical superiority.

Also tested: mixing v1 dynamics with v3 decoder → MSE 0.0112 (worse than v1's 0.0133).
**Latent spaces are model-private; the triad must be trained together.**

### ② Planning: one change took success from 56% → 100%

The bottleneck was not the world model but **action granularity** — v1 tried only 8 evenly spaced directions.

`continuous` candidate set (inspired by LingBot-World's action-conditioned control):

1. **Exact direction to target** (missing entirely in v1)
2. **±10°/±20° neighbors** of the aim direction (compensates the model's bias against straight rushes)
3. **16 continuous samples** (direction × magnitude 0.55-1.15×)
4. Momentum and brake candidates

| Strategy | Success (16 seeds) | Avg steps | Time |
|---|---|---|---|
| fixed8 (8 directions) | 9/16 (56%) | 27.1 | 37 s |
| **continuous** | **16/16 (100%)** | **26.3** | 40 s |

Imagination horizon 20 vs 12: identical (saturated). Kept at 12.

### ③ Language model: char 77.5M → BPE 123.4M

A from-scratch GPT (`tinygpt.py`, ~200 lines), two configurations:

| Version | Params | Vocab | Corpus | Role |
|---|---|---|---|---|
| char | 77.5M | ~5K chars | 20.4M chars | production (chat/perception/knowledge) |
| BPE | 123.4M | BPE 6400 | 20M tokens | validated alternative (better generalization) |

BPE migration pipeline: 54 MB Chinese corpus → 30M chars cleaned → 6400-token BPE (6.5 s) →
20M tokens pre-encoded → 123M pretrain (GPU 26 min) → SFT (3542 pairs).

**Honest comparison:**

| | 26M + hand-written | **123M + industrial** | char 77.5M (prod.) |
|---|---|---|---|
| Garbled output | severe | **none** | none |
| Tool decisions | n/a | **3/5 correct** | 100% (rule layer) |
| Multi-turn agentic | broken | unstable (repetition) | stable |

**Conclusion: BPE is an efficiency lever, not a quality lever.** Compression is real (55% fewer tokens
at equal context), but a small hand-written corpus only teaches memorization. Generalization comes from
**data scale**, not parameter count.

### ④ BPE tokenizer: from scratch, 100× faster

After two failed `pip install tokenizers`, we wrote our own (`bpe_tokenizer.py`, pure Python):

- Standard character-level BPE + ByteLevel compatible
- 6400 tokens (2053 chars + 4347 merges)
- **Incremental algorithm** (pair→position index + linked list + heap): 3M chars in **6.5 s** (naive scan: 10+ min)
- Chunked encoding + pre-encoded cache (`tokenize_corpus.py`)

**Vocabulary is domain-dependent (measured, two versions):**

| Training corpus | Compression | Learned tokens | Good for |
|---|---|---|---|
| Hand-written dialog (57K chars) | **2.22** | `世界模型` `语言模型` `小WM` | in-project dialog (high repetition) |
| Industrial (3M chars sampled) | **1.19** | `中国` `发展` `公司` `经济` | general Chinese (broad coverage) |

The shipped `bpe_6400.json` is the **industrial** version. Swapping vocabularies **invalidates all BPE
weights** (token ids change) and requires retraining 123M → SFT → fusion (~40 min).

### ⑤ Agent layer: event memory, parametric actions, remote tools

**WorldState** (inspired by LingBot-World 2.0's "in-world event evolution"):

The agent explicitly maintains a world copy — position/velocity/target/distance plus an **event stream**
(reached / bumped / failed / timeout). Previously state lived only in environment ground truth, so the
model could not know "I bumped into a wall" — that is reflex, not memory.

```
刚才发生了什么 → 最近一次是到达目标（36 步）。到过目标 1 次、撞墙 0 次、失败 0 次。
你撞墙了吗     → 撞过，一共 2 次。每次反弹回来我就换方向再试。
```

**Parametric tool calls** (MiniMind's Tool Call × LingBot's continuous actions):

```
Motion      2-D continuous acceleration        ← world model's native space
Interaction discrete tool selection            ← MiniMind's Tool Call
Parametric  [调用工具 world_act, direction=西北, magnitude=0.7]
```

Full chain: `_parse_tool` parses arguments → `execute_tool` translates to continuous control →
result is fed back → the model summarizes. 90 parametric trajectories in SFT data; 3/5 correct in tests
(**up/down still flips** — spatial direction mapping is hard for a char-level small model; production
falls back to a rule layer rather than adding data).

**Remote tools**: `register_http_tool("kb", "http://host/kb")` registers any HTTP endpoint as a tool,
overriding a local implementation of the same name, zero dependencies (stdlib `urllib`). Cross-process
calls to `api_server.py` verified.

### ⑥ Fusion: injecting the world model's latent into a 123M BPE model

The project's core fusion point: give the BPE language model a latent-state channel.

```
world model encode → z (32-D) → state_proj → added only at the [STATE] token embedding
                                             ↓
                              generation: "target is north, distance 0.7"
```

**Measured** (same env frames, BPE 123M vs char 77.5M):

| | Format | Hallucination | Distance error |
|---|---|---|---|
| **BPE 123M + state** | correct (`直线距离0.6个单位`) | **none** | 0.1-0.4 |
| char 77.5M + state | correct | **yes** (`西北大学附属中学建校于1952年…`) | 0.1-0.5 |

Real output (seed 3003, true distance 0.69):

```
你看到了什么？        -> 画面里有个亮点是我自己（正往正西移动），在偏上右侧；还有一个灰色圆环是目标点，在偏上右侧。
目标离你有多远？       -> 直线距离0.6个单位。
你现在在往哪边走？     -> 西北方向，距离约0.6。
你和目标的方位关系？   -> 往正南边走，大约0.6个单位远。
```

Key design: **the perception channel isolates dialog history.** Multi-turn context pushes the `[STATE]`
token out of position, breaking injection (measured: after 4 knowledge questions, "what do you see?"
answered with an American fantasy film). Fix: `_generate(..., use_ctx=False)` — perception is an
independent channel.

---

## Experiment log: wins and losses

All A/B runs use 16 seeds from identical starting states unless noted.

| Experiment | Result | Verdict |
|---|---|---|
| Continuous action candidates | 9/16 → **16/16** | ✅ action granularity was the real bottleneck |
| Horizon 20 vs 12 | identical | ➖ saturated, kept 12 |
| v2 mixed-strategy data | 9/16 → 10/16 | ❌ data is not the bottleneck |
| v3 state head (navigation) | 12/12 → 8/12 | ❌ 10× less accurate than pixel decode |
| v3 state head (reconstruction) | MSE 0.0133 → **0.0018** | ✅ clearly better decoder |
| v3 decode (8 seeds) | 7/8 vs v1 8/8 | ❌ larger multi-step drift |
| v1 dynamics + v3 decoder | MSE 0.0112 vs v1 0.0133 | ❌ latent spaces incompatible |
| BPE vs char (same 26M/corpus) | efficiency up, quality down | ❌ needs industrial-scale corpus |
| 123M industrial BPE | no garbling + 3/5 tool decisions | ✅ generalization from data scale |
| BPE 123M + state injection | zero hallucination, err 0.1-0.4 | ✅ fusion works |
| GPU batch 48 vs 128 | 629 ms vs 13642 ms | ❌ ROCm large-batch pathology (not compute) |

---

## Training recipe

```bash
# World model (CPU, 4 min)
python train.py --episodes 400 --epochs 20

# Perception language model (char, GPU 12 min)
python gen_state_data.py --episodes 150 --out state_qa.jsonl
python train_lm_state.py --resume lm_final2.pt --state-data state_qa.jsonl \
    --plain-corpus corpus_dialog.txt --epochs 4 --out lm_state.pt

# BPE pipeline (GPU ~40 min)
python download_corpus.py
python build_final_corpus.py
python bpe_tokenizer.py --train --vocab-size 6400 --corpus corpus_bpe_sample.txt
python tokenize_corpus.py --corpus corpus_v3.txt --out tokens_v3.npy --max-chars 25000000
python corpus_sft.py --out sft_data_big.jsonl
python train_bpe.py --tokens tokens_v3.npy --d-model 768 --n-layers 16 --n-heads 12 \
    --epochs 3 --batch-size 48 --out lm_bpe_big.pt
python train_sft.py --data sft_data_big.jsonl --resume lm_bpe_big.pt \
    --out lm_bpe_big_sft.pt --epochs 3 --bpe bpe_6400.json --block-size 512

# Fusion (BPE 123M + latent injection, GPU 11 min)
python train_lm_state.py --bpe bpe_6400.json --state-dim 32 --seq-len 128 \
    --resume lm_bpe_big_sft.pt --state-data state_qa.jsonl --plain-corpus "" \
    --epochs 3 --batch-size 48 --out lm_state_bpe.pt
```

**Hardware**: RX 6800 16 GB (ROCm 7.16 / torch 2.13) or any 12 GB+ NVIDIA card.
The world model also trains on CPU in 4 minutes.

**ROCm notes**: keep batch at 48 — batch 96 is 9× slower, batch 128 is 22× slower (allocator pathology,
not compute). Use fp16 autocast; fp32 on RDNA2 does not use matrix acceleration (0.5 TFLOPS vs 18 theoretical).

---

## Pitfalls (12 real ones)

1. **Decoder collapses to black under MSE.** Agent pixels are ~1% of the image; "all black" has lower loss
   than learning to draw the dot. → foreground-weighted BCE (weight 15).
2. **Joint training avalanche.** Dynamics gradients flow back into the encoder, representation drifts,
   dyn loss goes from 1e-4 to 1.2e8. → `z.detach()` + lr 5e-4 + grad clip.
3. **Saving the last epoch saves the crash.** Diverged at epoch 23. → save best only.
4. **Wall-bump detection missed a coefficient**, 24 false positives in 36 steps. Real physics:
   `v' = 0.8v + 0.12a`, `pos' = pos + v' * 0.10`. → **if the metric is wrong, don't trust the numbers.**
5. **BPE vocabulary id allocation bug**: `len(vocab)+i` grows as vocab grows, last id reached 10745. → `base + i`.
6. **collate block misalignment**: padded to 256 while `block_size` was 255 → positions out of range.
7. **`'\n'` is id 0 in the BPE vocab**, so `stop_ids={0}` truncates generation at the first token
   (symptom: empty output). → truncate after decoding in BPE mode.
8. **The probe that reported "GPU is 20× slower" was wrong**: it used BLOCK=512 while real training uses 256.
   → **align the probe with the real config before drawing conclusions.**
9. **ROCm large-batch pathology**: 629 ms/step at batch 48 vs 13642 ms at batch 128 (22×, while ms/sample grows).
10. **state loss gradients wrecked the encoder** (v3): dyn loss exploded to 11496. → state head must use `z.detach()`.
11. **Sequence padding wasted 4× compute**: state QA text is 60-80 tokens but was padded to 511.
    → `--seq-len 128` took training from 34 min/epoch to 3.5 min/epoch.
12. **Strict `load_state_dict` breaks compatibility**: after adding `state_head`, old v1 weights fail to load.
    → load by shape match.

---

## Learning path

Read in dependency order; every file runs standalone.

| # | File | What you learn | Time |
|---|---|---|---|
| 1 | `env.py` (200 lines) | building a minimal physics environment | 20 min |
| 2 | `model.py` + `train.py` | encoder / dynamics / decoder triad | 1 h |
| 3 | `demo.py` | 1-step prediction, open-loop rollout, MPC | 30 min |
| 4 | `agent.py` → `_run_navigation` | MPC + PD hierarchical control, continuous candidates | 1 h |
| 5 | `tinygpt.py` (200 lines) | GPT from scratch: attention, residuals, generation | 1 h |
| 6 | `bpe_tokenizer.py` | BPE training and the incremental algorithm | 1 h |
| 7 | `train_lm_state.py` | how latent injection is trained | 1 h |
| 8 | `agent.py` (full) | tool calls, event memory, 4-layer fallback | 2 h |

Prerequisites: basic PyTorch (can write an `nn.Module`), comfortable with numpy.
No distributed training or RL experience required.

---

## Relationship to related work

| Project | What it does | How this differs |
|---|---|---|
| [MiniMind](https://github.com/jingyaogong/minimind) | trains small LMs from scratch (26M-108M) | LM config follows its recipe; this adds a world model and an agent loop |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | minimal GPT training | this is even smaller and hand-written (~200 lines), with perception injection |
| [LingBot-World](https://github.com/robbyant/lingbot-world) | video-generative world model (14B) | 5 orders of magnitude apart; borrows action-conditioned control and event evolution |
| [World Models](https://arxiv.org/abs/1803.10122) / Dreamer | latent world models | an isomorphic minimal implementation trainable on CPU |

**Only care about language models?** Go to MiniMind; it's more complete.
**Only care about world models?** Go to DreamerV3; it's stronger.
**Want to see how the three fit into one runnable system?** That is this project's only reason to exist.

---

## Limitations

- The world model handles only a 64×64 grayscale point-mass environment; it does not generalize to real images
- **One-step imagined frames are blurry** (MSE regression averages modes). v3's decoder is clearly sharper
  ([comparison](docs/images/demo_v3.png)) but its multi-step dynamics are worse — each excels at something
  different, and the default picks v1 for navigation stability
- The 123M LM cannot compete with any open-source small LLM; multi-turn agentic behavior degrades into repetition
- State injection is limited by the 32-D latent (distance error 0.1-0.4); replacing pixel decoding needs a larger latent
- Parametric tool calls still flip up/down; production relies on a rule layer
- Training data is hand-written plus public corpora (~30M chars) — not enough to make the model genuinely capable

## Roadmap

- [ ] Expand SFT to 20K pairs (pipeline ready, data-only)
- [ ] Full 20M-token pretrain (currently 10M)
- [ ] Keypoint heatmap instead of decode+argmax (target state accuracy 0.02)
- [ ] Learn a real policy network to replace MPC (Dreamer route)
- [ ] Multi-room environment + long-term memory

## Acknowledgements

[MiniMind](https://github.com/jingyaogong/minimind) (SFT configuration, Tool Call design),
[LingBot-World](https://github.com/robbyant/lingbot-world) (action-conditioned control, event evolution),
[World Models](https://arxiv.org/abs/1803.10122) (the triad architecture).

Full technical log (all experiments, including failures) in [`docs/ARCHIVE.md`](docs/ARCHIVE.md) (Chinese).

## License

[MIT](LICENSE)
