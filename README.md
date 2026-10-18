# mini-world-model

**从零训练的世界模型 × 语言模型 × 具身 Agent —— 单卡可复现的完整闭环**

一个 6.7K 行、纯 PyTorch 实现的最小具身智能系统：像素感知 → 潜空间动力学 → MPC 规划 → 动作执行 → 事件记忆 → 语言交互。
所有模型**从零训练**（不 import transformers），单张 RX 6800 / RTX 3090 可完整复现，全部实测数据公开——**包括失败的实验**。

```
导航成功率   9/16 → 16/16      世界模型重建   MSE 0.0018（v3）/ 0.0133（v1）
BPE 压缩率   2.22 字符/token    感知问答      零幻觉，距离误差 0.1-0.4
语言模型     char 77.5M → BPE 123.4M        训练速度  123M @ 629 ms/step（RX 6800）
```

---

## 这是什么 / 不是什么

**是**：一个把「世界模型 + 语言模型 + agent」三层完整打通的**教学级实现**。
每一层都不调库、不套壳，代码可读、数据可复现、结论有实测。

**不是**：不是产品，不是 SOTA，不能处理真实图像（输入只有 64×64 灰度图），
语言能力远不及任何开源小 LLM。它的价值在于**把整个系统摊开给你看**。

如果你符合以下任一条，这个仓库对你有用：

- 想搞懂「世界模型到底在预测什么」，而不是只看论文图
- 想从零搭一个 agent，但不想直接抄 LangChain
- 想知道小模型在什么条件下真的能用、「加参数」到底有没有用
- 想找一份**诚实**的实验记录（大量项目只报喜，这里报忧）

---

## 架构

```
真实环境 PointMassEnv（64×64 灰度，2 维加速度动作）
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  ① 世界模型（~1.5M 参数，CNN + MLP）                       │
│     观测 ──Encoder CNN──> z_t（32 维潜状态）               │
│     动作 a_t ─────────> Dynamics MLP ──> ẑ_{t+1}          │
│              ┌───────────┴───────────┐                   │
│              ▼                       ▼                   │
│        Decoder CNN              State Head（v3 实验）      │
│        想象的下一帧              z → pos/vel/dist        │
└─────────────────────────────────────────────────────────┘
        │ z_t
        ▼
┌─────────────────────────────────────────────────────────┐
│  ② 规划与控制                                             │
│     MPC（在世界模型里想象 12 步）──> 选动作 ──> PD 控制    │
│     候选集：朝目标精确方向 + ±10°/±20° 邻域 + 16 连续采样  │
└─────────────────────────────────────────────────────────┘
        │ 执行 + 事件
        ▼
┌─────────────────────────────────────────────────────────┐
│  ③ Agent 层（agent.py 单入口）                            │
│     WorldState：位置/速度/目标/距离 + 事件流               │
│     事件：到达 / 撞墙 / 失败 / 超时 —— 可被语言层引用      │
│     工具：world_perceive / world_navigate / world_act     │
│           world_imagine / kb_query / calc                 │
│     远程工具：register_http_tool() 任意 HTTP 端点          │
└─────────────────────────────────────────────────────────┘
        │ z_t 注入 [STATE] token
        ▼
┌─────────────────────────────────────────────────────────┐
│  ④ 语言模型（char 77.5M / BPE 123.4M，从零训练）          │
│     感知问答：潜状态注入 → "目标在正北，直线距离 0.7"       │
│     对话/知识：四层兜底（查表 → 检索 → 生成 → 说不知道）    │
│     工具调用：[调用工具 world_act, direction=西北]        │
└─────────────────────────────────────────────────────────┘
```

调用方式三种：**CLI / HTTP API / MCP Server**。

---

## 核心结果

| 能力 | 指标 | 说明 |
|---|---|---|
| **导航成功率** | **16/16（100%）** | 16 seed，continuous 候选；fixed8 基线仅 9/16 |
| 导航效率 | 26.3 步 / 40 秒 | 平均，含规划开销 |
| 世界模型重建 | MSE **0.0018**（v3）/ 0.0133（v1） | 前景加权 BCE，64×64 灰度 |
| 开环想象 | 8 步内误差 < 0.1 | 30 步发散到 ~0.8（世界模型通病） |
| **感知问答** | **零幻觉**，距离误差 0.1-0.4 | 潜状态注入，BPE 123M 版 |
| 对话/知识 | 11 项全部正常 | 四层兜底，常识库 669 条 |
| 工具调用 | 3/5 自主决策正确 | "23×7→calc"、"长城→kb_query" |
| BPE 压缩率 | **2.22 字符/token** | 手写语料；工业语料 1.19 |
| BPE 训练速度 | 300 万字符 **6.5 秒** | 增量算法（全扫描版要 10 分钟） |
| 训练速度 | 123M @ **629 ms/step** | RX 6800，fp16，batch 48，block 256 |

---

## 5 分钟跑通

```bash
# 0. 依赖（torch 装 CPU 版即可跑世界模型）
pip install -r requirements.txt

# 1. 训世界模型（CPU 约 4 分钟）——没有权重时会给出明确指引
python train.py --episodes 400 --epochs 20

# 2. 三个基础实验：一步预测 / 开环想象 / MPC 规划
python demo.py          # 产出 demo_result.png

# 3. CLI 交互（需要语言模型权重，见训练配方；缺失时自动降级）
python agent.py "你看到了什么？"     # 感知（真跑世界模型编码器）
python agent.py "去目标那儿"         # 导航（MPC 规划）
python agent.py "光速是多少？"       # 常识检索
python agent.py "23 乘以 7"          # 符号计算
```

**HTTP API**（`/` 端点自描述所有接口）：

```bash
python api_server.py --port 8399 &
curl http://127.0.0.1:8399/                                    # 列出全部端点
curl -X POST http://127.0.0.1:8399/reply -d '{"text":"你好"}'
curl -X POST http://127.0.0.1:8399/task  -d '{"task":"去目标那儿"}'
curl -X POST http://127.0.0.1:8399/calc  -d '{"expression":"23*7"}'
curl http://127.0.0.1:8399/state                               # 位置/事件/步数
```

**A/B 评测框架**（任何改动先过它，别看单个 seed）：

```bash
python eval_world.py   --seeds 16                  # 世界模型对比
python eval_planner.py --seeds 16 --horizons 20    # 规划策略对比
python verify_state_bpe.py --ckpt lm_state_bpe.pt  # 感知问答对比
python bench_probe.py                              # GPU 性能探针
```

---

## 模块详解

### ① 世界模型：三件套 + 三代实验

经典结构（与 World Models / Dreamer 同构，规模小 5 个数量级）：

| 组件 | 本实现 | DreamerV3 级别 |
|---|---|---|
| 观测 | 64×64 灰度 | 720p 视频 |
| 编码器 | 4 层 CNN | ViT / 3D VAE |
| 动力学 | 128 隐层 MLP | 数亿参数 Transformer |
| 训练数据 | 2.4 万帧 rollout | 百万小时视频 |

**三个实验验证它真的学到了东西**：

1. **一步预测**：想象帧里 agent 位置、形状都对上 → encoder+dynamics 学到了视觉动态
2. **开环想象**：纯潜空间迭代 30 步，前 8 步误差 < 0.1，10 步后发散
3. **MPC 规划**：在想象里试错选动作，配合 continuous 候选达到 100%

**两代不成功但有价值的尝试**：

| 版本 | 改动 | 结果 |
|---|---|---|
| v2 | 50% 随机 + 50% MPC 数据，数据量翻倍 | 导航 9/16 → 10/16（+6%，噪声级）→ **数据不是瓶颈** |
| v3 | 加 state head（z → pos/vel/target/dist） | 重建 MSE **0.0018**（比 v1 好 7 倍），但 MPC 走状态头精度差 10 倍 |

**v3 的完整结论**（三配置各 8 seed）：

| 配置 | 导航成功率 | 说明 |
|---|---|---|
| v1 + decode+argmax | **8/8** | 当前默认 |
| v3 + state head | 5/8 | 状态头 RMSE 0.178 vs 像素 1/64≈0.016 |
| v3 + decode+argmax | 7/8 | 单步重建更准，但多步动力学累积误差更大 |

**结论：单步重建最优 ≠ 多步规划最优。** v3 的 decoder 明显更好（MSE 好 7 倍），
但它的 dynamics 在 12 步想象窗口上更容易发散，所以 MPC 反而更差。
默认选 v1 是工程取舍，不是技术优劣。

### ② 规划：一次改动把成功率从 56% 拉到 100%

问题不在世界模型，在**动作候选太粗**——v1 只试 8 个等分方向，经常错过最优动作。

`continuous` 候选集（借鉴 LingBot-World 的 action-conditioned 思想）：

1. **朝目标的精确方向**（v1 完全没有这个候选）
2. 瞄准方向 **±10°/±20° 邻域**（补偿世界模型对直冲动作的系统偏置）
3. **16 个连续采样**（方向 × 幅度 0.55-1.15×）
4. 惯性 + 减速候选

| 策略 | 成功率（16 seed） | 平均步数 | 耗时 |
|---|---|---|---|
| fixed8（8 个固定方向） | 9/16（56%） | 27.1 | 37s |
| **continuous** | **16/16（100%）** | **26.3** | 40s |

想象窗口 20 步 vs 12 步效果相同（已饱和），保持 12。

### ③ 语言模型：char 77.5M → BPE 123.4M

从零训练的 GPT（`tinygpt.py`，约 200 行），两代配置：

| 版本 | 参数 | 词表 | 语料 | 用途 |
|---|---|---|---|---|
| char | 77.5M | 字符级 ~5K | 2040 万字符 | 生产配置（对话/感知/知识） |
| BPE | 123.4M | BPE 6400 | 2000 万 token | 已验证备选（泛化更好） |

**BPE 迁移完整链路**（「先修架构再扩规模」的实践）：

```
download_corpus.py   54.4MB 中文语料（23 秒，hf-mirror）
  → corpus_v3.txt    3000 万字符（清洗去重）
  → BPE 6400 词表    6.5 秒（增量算法，100 倍提速）
  → tokens_v3.npy    2000 万 token（预编码，17 分钟一次性）
  → 123M pretrain    16 层/768，1000 万 token × 3ep，GPU 26 分钟
  → SFT              3542 条，loss 3.43 → 1.40
```

**效果对比（诚实版）**：

| | 26M + 手写语料 | **123M + 工业语料** | char 77.5M（生产） |
|---|---|---|---|
| 乱码 | 严重 | **无** | 无 |
| 工具决策直测 | 无 | **3/5 正确** | 路由层 100% |
| agentic 多轮 | 崩 | 不稳（重复退化） | 稳定 |

**结论：BPE 是效率杠杆，不是质量杠杆。** 压缩率 2.22 字符/token（同等上下文 token 省 55%）是真的，
但小语料喂大模型只能背诵；泛化能力来自**数据规模**，不是参数规模。

### ④ BPE 分词器：从零实现 + 100 倍提速

`pip install tokenizers` 两次失败后干脆自己写（`bpe_tokenizer.py`，纯 Python 无依赖）：

- 标准字符级 BPE 训练 + ByteLevel 兼容
- 6400 词表（2053 字符 + 4347 merges）
- **增量算法**：pair→位置索引 + 链表 + 堆，300 万字符 **6.5 秒**（全扫描版 10 分钟+）
- 分块编码 + 预编码缓存（`tokenize_corpus.py`）

学到的 token：`世界模型` `语言模型` `小WM` `目标` `规划` `画面`——
结构化对话语料重复度高，BPE 能合并成长片段，这是压缩率超过 MiniMind（1.5-1.7）的原因。

### ⑤ Agent 层：事件记忆 + 参数化动作 + 远程工具

**WorldState（借鉴 LingBot-World 2.0 的「世界内事件演化」）**：

agent 显式维护世界副本——位置/速度/目标/距离 + **事件流**（到达/撞墙/失败/超时）。
此前状态散落在环境真值里，模型无从知道「我撞过墙」——那是反应不是记忆。

```
刚才发生了什么 → 最近一次是到达目标（36 步）。到过目标 1 次、撞墙 0 次、失败 0 次。
你撞墙了吗     → 撞过，一共 2 次。每次反弹回来我就换方向再试。
```

**参数化工具调用（MiniMind Tool Call × LingBot 连续动作的交汇）**：

```
移动动作    2 维连续加速度           ← 世界模型的原生动作空间
交互动作    工具调用（离散选择）      ← MiniMind 的 Tool Call
参数化动作  [调用工具 world_act, direction=西北, magnitude=0.7]  ← 融合
```

链路完整：`_parse_tool` 解析参数 → `execute_tool` 翻译成连续控制 → 结果回灌 → 模型总结。
SFT 数据 90 条参数化轨迹。实测 3/5 正确（**上下方向仍会反转**——空间方向映射是 char-level
小模型的硬骨头，与 Δz 实验结论一致，解法是规则层兜底而不是加数据）。

**远程工具**：`register_http_tool("kb", "http://host/kb")` 把任意 HTTP 端点注册成工具，
同名可覆盖本地实现，零依赖（stdlib urllib）。实测跨进程调用 `api_server.py` 端点通。

### ⑥ 融合：把世界模型的潜状态塞进 123M BPE 模型

这是整个项目**最核心的融合点**：给 BPE 大模型装一条潜状态注入通道。

```
世界模型 encode → z（32 维）→ state_proj → 只在 [STATE] token 位置加到 embedding
                                             ↓
                              语言模型生成："目标在正北，直线距离 0.7"
```

**实测**（同一批 env 帧，BPE 123M vs char 77.5M）：

| | 格式 | 幻觉 | 距离误差 |
|---|---|---|---|
| **BPE 123M + state** | 完美（"直线距离0.6个单位"） | **无** | 0.1-0.4 |
| char 77.5M + state | 有 | **有**（"西北大学附属中学建校于1952年…"） | 0.1-0.5 |

真实输出（seed 3003，真实距离 0.69）：

```
你看到了什么？        -> 画面里有个亮点是我自己（正往正西移动），在偏上右侧；还有一个灰色圆环是目标点，在偏上右侧。
目标离你有多远？       -> 直线距离0.6个单位。
你现在在往哪边走？     -> 西北方向，距离约0.6。
你和目标的方位关系？   -> 往正南边走，大约0.6个单位远。
```

关键设计：**感知通道隔离对话历史**。多轮上下文会挤掉 `[STATE]` 的位置导致注入失效
（实测：先问 4 个知识问题再问"你看到了什么"，答成美国奇幻电影）。
修法：`_generate(..., use_ctx=False)`——感知是不依赖历史的独立通道。

---

## 实验记录：赢的和输的

诚实是这个仓库的底线。全部 A/B 实验（16 seed，同起点，除非注明）：

| 实验 | 结果 | 结论 |
|---|---|---|
| 连续动作候选 MPC | 9/16 → **16/16** | ✅ 规划粒度是真瓶颈 |
| 想象窗口 20 vs 12 步 | 相同 | ➖ 已饱和，保持 12 |
| v2 混合策略数据 | 9/16 → 10/16 | ❌ 数据不是瓶颈 |
| v3 状态头（导航） | 12/12 → 8/12 | ❌ 精度输给像素解码 10 倍 |
| v3 状态头（重建） | MSE 0.0133 → **0.0018** | ✅ 解码器明显更强 |
| v3 decode（8 seed） | 7/8 vs v1 8/8 | ❌ 多步动力学累积误差更大 |
| v1 dynamics + v3 decoder | MSE 0.0112 vs v1 0.0133 | ❌ 潜空间不兼容，三件套必须成套训 |
| BPE vs char（同 26M/同语料） | 效率赢、质量输 | ❌ 需要工业语料规模 |
| 123M 工业语料 BPE | 无乱码 + 3/5 工具正确 | ✅ 泛化来自数据规模 |
| BPE 123M + state 注入 | 零幻觉、误差 0.1-0.4 | ✅ 融合链路成立 |
| GPU batch 48 vs 128 | 629ms vs 13642ms | ❌ ROCm 大 batch 灾难（非算力） |

---

## 训练配方（完整复现）

```bash
# 世界模型（CPU 4 分钟）
python train.py --episodes 400 --epochs 20

# 感知语言模型（char 版，GPU 12 分钟）
python gen_state_data.py --episodes 150 --out state_qa.jsonl
python train_lm_state.py --resume lm_final2.pt --state-data state_qa.jsonl \
    --plain-corpus corpus_dialog.txt --epochs 4 --out lm_state.pt

# BPE 全链路（GPU 约 40 分钟）
python download_corpus.py                                         # 54MB 语料
python build_final_corpus.py                                      # 清洗 → 3000 万字符
python bpe_tokenizer.py --train --vocab-size 6400 --corpus corpus_bpe_sample.txt
python tokenize_corpus.py --corpus corpus_v3.txt --out tokens_v3.npy --max-chars 25000000
python corpus_sft.py --out sft_data_big.jsonl                     # 3542 条 SFT
python train_bpe.py --tokens tokens_v3.npy --d-model 768 --n-layers 16 --n-heads 12 \
    --epochs 3 --batch-size 48 --out lm_bpe_big.pt
python train_sft.py --data sft_data_big.jsonl --resume lm_bpe_big.pt \
    --out lm_bpe_big_sft.pt --epochs 3 --bpe bpe_6400.json --block-size 512

# 融合（BPE 123M + 潜状态注入，GPU 11 分钟）
python train_lm_state.py --bpe bpe_6400.json --state-dim 32 --seq-len 128 \
    --resume lm_bpe_big_sft.pt --state-data state_qa.jsonl --plain-corpus "" \
    --epochs 3 --batch-size 48 --out lm_state_bpe.pt
```

**硬件**：RX 6800 16GB（ROCm 7.16 / torch 2.13）或任意 12GB+ NVIDIA 卡。
CPU 也能跑世界模型（4 分钟）。

**ROCm 用户注意**：batch 固定 48——实测 batch 96 慢 9 倍、128 慢 22 倍（显存管理劣化，
不是算力问题）。必须 fp16 autocast，fp32 在 RDNA2 上不走矩阵加速（0.5 TFLOPS vs 理论 18）。

---

## 踩坑记录（精选 12 条，全部真实撞过）

1. **decoder 被 MSE 欺骗坍缩成全黑**。agent 像素只占 1%，"全黑"的 loss 比学画亮点还低。
   → 前景加权 BCE（权重 15）。
2. **联合训练雪崩**。dynamics 梯度反传改 encoder，表征漂移 → dyn loss 从 1e-4 冲到 1.2 亿。
   → `z.detach()` 解耦 + lr 5e-4 + 梯度裁剪。
3. **保存最后一轮 = 保存崩溃权重**。第 23 轮发散。→ 只存历史最佳。
4. **撞墙检测公式漏系数**，36 步误报 24 次。真实物理是 `v' = 0.8v + 0.12a`、`pos' = pos + v'*0.10`。
   → **指标算错就别信，先核对公式**。
5. **BPE 词表 id 分配错误**：`len(vocab)+i` 循环里 len 在涨，最后一轮 id 冲到 10745 越界。
   → `base + i`。
6. **collate block 没对齐**：pad 到 256 而 block_size 是 255，positions 越界。→ `block_size - 1`。
7. **`'\n'` 在 BPE 词表里是 id 0**，`stop_ids={0}` 把生成立即掐断（表现为"生成为空"）。
   → BPE 模式解码后按字符截断。
8. **探针测出"GPU 慢 20 倍"是假的**：探针 BLOCK=512 而真实训练用 256，序列长差一倍、attention 差 4 倍。
   → **探针必须先对齐真实配置**。
9. **ROCm 大 batch 灾难**：batch 48 = 629ms/step，batch 128 = 13642ms/step（22 倍劣化，ms/样本还涨）。
   → batch 固定 48，有效 batch 靠累积。
10. **state loss 梯度冲垮 encoder**（v3）：dyn loss 爆到 11496。→ 状态头必须 `z.detach()`。
11. **序列 pad 浪费 4 倍算力**：state QA 文本只有 60-80 token，却 pad 到 511（模型 block_size）。
    → `--seq-len 128`，训练从 34 分钟/epoch 降到 3.5 分钟/epoch。
12. **严格 `load_state_dict` 会破兼容**：model.py 加了 state_head 后，旧的 v1 权重严格加载失败。
    → 按形状匹配加载（只复制匹配的 key）。

---

## 学习路径

按依赖顺序读，每个文件都能独立跑：

| 顺序 | 文件 | 学到什么 | 耗时 |
|---|---|---|---|
| 1 | `env.py`（200 行） | 一个最小物理环境怎么造 | 20 分钟 |
| 2 | `model.py` + `train.py` | 编码器/动力学/解码器三件套 | 1 小时 |
| 3 | `demo.py` | 一步预测/开环想象/MPC 三个实验 | 30 分钟 |
| 4 | `agent.py` 的 `_run_navigation` | MPC + PD 分层控制、连续动作候选 | 1 小时 |
| 5 | `tinygpt.py`（200 行） | 从零 GPT：attention/残差/生成 | 1 小时 |
| 6 | `bpe_tokenizer.py` | BPE 训练与增量算法 | 1 小时 |
| 7 | `train_lm_state.py` | 潜状态注入怎么训 | 1 小时 |
| 8 | `agent.py` 全文 | 工具调用/事件记忆/四层兜底 | 2 小时 |

前置要求：PyTorch 基础（会写 nn.Module）、能读 numpy。不需要分布式/RL 经验。

---

## 与相关项目的关系

| 项目 | 它做什么 | 本项目的差异 |
|---|---|---|
| [MiniMind](https://github.com/jingyaogong/minimind) | 从零训小语言模型（26M-108M） | 语言模型部分参考其配置；额外有世界模型和 agent 闭环 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 最小 GPT 训练 | 本项目是更小的纯手写实现（200 行），且带感知注入 |
| [LingBot-World](https://github.com/robbyant/lingbot-world) | 视频生成型世界模型（14B） | 尺度差 5 个数量级；借鉴其 action-conditioned 控制与事件演化思想 |
| [World Models](https://arxiv.org/abs/1803.10122) / Dreamer | 潜空间世界模型 | 同构的极小实现，CPU 可训 |

**只关心语言模型**：去看 MiniMind，比这里完整。
**只关心世界模型**：去看 DreamerV3 代码，比这里强。
**想看三者怎么拼成一个能跑的系统**：这是本项目唯一的立足点。

---

## 局限（写清楚，不藏）

- 世界模型只处理 64×64 灰度点质量环境，不能泛化到真实图像
- **一步想象的解码帧偏模糊**（MSE 回归的均值化效应）。v3 的 decoder 明显更清晰
  （MSE 好 7 倍，[对比图](docs/images/demo_v3.png)），但多步动力学不如 v1——两个版本各有所长，
  默认选 v1 是为了导航稳定。**不能混用**：试过 v1 dynamics + v3 decoder，MSE 0.0112
  （比 v1 的 0.0133 还差）——潜空间是模型私有的，三件套必须成套训练
- 语言模型 123M，能力远不及任何开源小 LLM；agentic 多轮会重复退化
- 状态注入精度受限于 32 维 z（距离误差 0.1-0.4），要替代像素解码需更大潜空间
- 参数化工具调用的空间方向映射不稳（上下会反转），生产靠规则层兜底
- 训练数据是手写 + 公开语料，规模 3000 万字符，不足以让模型真正"聪明"

## 路线

- [ ] SFT 扩到 2 万条（管线已就绪，纯加数据）
- [ ] 全量 2000 万 token pretrain（当前只用了 1000 万）
- [ ] 关键点热图替代 decode+argmax（把状态精度做到 0.02）
- [ ] 学一个真正的策略网络替代 MPC（Dreamer 路线）
- [ ] 多房间环境 + 长期记忆

## 引用 / 致谢

参考并致谢：[MiniMind](https://github.com/jingyaogong/minimind)（SFT 配置与 Tool Call 设计）、
[LingBot-World](https://github.com/robbyant/lingbot-world)（action-conditioned 控制、事件演化思想）、
[World Models](https://arxiv.org/abs/1803.10122)（三件套架构）。

完整技术日志（含全部实验过程与失败记录）见 [`docs/ARCHIVE.md`](docs/ARCHIVE.md)。

## License

[MIT](LICENSE)
