"""训练 L2 半耦合模型：世界模型潜状态 → 语言模型 token 注入。

数据混合：
  - state_qa.jsonl（"潜状态→语言描述"配对，每条带 32 维 state）
  - 普通语料（corpus_final.txt 抽样，防灾难性遗忘，state 填零——
    没有 [STATE] token 的样本 mask 全零，state 自动失效）

warm start：从 lm_final2.pt 继承，词表 +1（[STATE] token）时
复制旧 embedding 行、追加新行，保留已学到的词表知识。

用法：python train_lm_state.py --epochs 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygpt import TinyGPT, count_params
from tokenizer import load_tokenizer

STATE_TOKEN = "\uE000"
BLOCK = 96


def load_resume(path):
    ck = torch.load(path, map_location="cpu")
    return ck["stoi"], ck["itos"], ck["model"]


def build_state_samples(path, stoi, tok=None, block=512, state_dim=None):
    """state QA：每条样本 (input, target, state)。

    tok 非空时用 BPE 编码；state_dim 非空时从 state 截取前 N 维
    （64 维数据取前 32 = 纯 z，丢弃 Δz 部分）。
    """
    xs, ys, states = [], [], []
    space = tok.eot_id if tok is not None else stoi[" "]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if tok is not None:
                ids = tok.encode(obj["text"])
            else:
                ids = [stoi[c] for c in obj["text"] if c in stoi]
            if len(ids) < 8:
                continue
            ids = ids[:block]
            inp, tgt = ids[:-1], ids[1:]
            inp = inp + [space] * (block - 1 - len(inp))
            tgt = tgt + [space] * (block - 1 - len(tgt))
            st = obj["state"]
            if state_dim:
                st = st[:state_dim]
            xs.append(inp)
            ys.append(tgt)
            states.append(st)
    return xs, ys, states


def build_plain_samples(path, stoi, char_budget, state_dim, repeat=1):
    """普通语料：切窗样本，state 填零。repeat 用于小语料扩量（对话语料需要）。"""
    if not os.path.exists(path):
        return [], [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng = np.random.default_rng(0)
    rng.shuffle(lines)
    text, n = [], 0
    for line in lines:
        text.append(line)
        n += len(line)
        if n >= char_budget:
            break
    text = "".join(text) * repeat
    data = [stoi[c] for c in text if c in stoi]
    n_win = max(1, (len(data) - 1) // BLOCK)
    xs = [data[i * BLOCK : i * BLOCK + BLOCK][:-1] for i in range(n_win)]
    ys = [data[i * BLOCK + 1 : i * BLOCK + BLOCK] for i in range(n_win)]
    states = [[0.0] * state_dim for _ in range(n_win)]
    return xs, ys, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default="lm_final2.pt")
    parser.add_argument("--state-data", type=str, default="state_qa_motion.jsonl")
    parser.add_argument("--plain-corpus", type=str, default="corpus_final.txt")
    parser.add_argument("--plain-budget", type=float, default=2.7, help="普通语料字符预算（百万）")
    parser.add_argument("--repeat", type=int, default=1, help="普通语料重复次数（小语料扩量用）")
    parser.add_argument("--out", type=str, default="lm_state_motion.pt")
    parser.add_argument("--bpe", type=str, default=None, help="BPE 词表路径（BPE 模式）")
    parser.add_argument("--state-dim", type=int, default=None, help="state 截取维度（64 维取 32）")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="数据序列长度（默认用模型 block_size；state QA 文本短，"
                             "截到 128 可省 16 倍 attention 算力）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 词表：BPE 模式从词表文件 + 特殊 token；char 模式继承 + 追加
    bpe_tok = load_tokenizer(args.bpe) if args.bpe else None
    old_state = None
    if bpe_tok is not None:
        stoi = bpe_tok.bpe.vocab
        itos = None
        if STATE_TOKEN not in stoi:
            stoi[STATE_TOKEN] = len(stoi)
            bpe_tok.specials[STATE_TOKEN] = stoi[STATE_TOKEN]
        state_token_id = stoi[STATE_TOKEN]
        ck_bpe = torch.load(args.resume, map_location="cpu")  # warm start 源
        old_state = ck_bpe["model"]
        ck_cfg = ck_bpe["config"]
        print(f"BPE vocab: {len(stoi)} (state token id = {state_token_id}, block {ck_cfg['block_size']})")
    else:
        stoi, itos, old_state = load_resume(args.resume)
        if STATE_TOKEN not in stoi:
            stoi[STATE_TOKEN] = len(stoi)
            itos[len(itos)] = STATE_TOKEN
        state_token_id = stoi[STATE_TOKEN]
        print(f"vocab: {len(stoi)} (state token id = {state_token_id})")

    # 2. 数据（state 维度从数据探测；无 state 数据时 state_dim=0 训纯文本模型）
    # block 必须与模型 block_size 一致：BPE 用 ck_cfg（255/511），char 用 BLOCK-1（95）。
    # 踩坑：不传 block 时默认 512 → 序列 pad 到 511 → BPE 模型 position 越界
    # 优化：state QA 文本短（≤80 token），--seq-len 截到 128 省 16 倍 attention
    blk = ck_cfg["block_size"] if bpe_tok is not None else BLOCK - 1
    if args.seq_len:
        blk = min(blk, args.seq_len)
    sx, sy, ss = (build_state_samples(args.state_data, stoi, tok=bpe_tok,
                                      block=blk, state_dim=args.state_dim)
                  if args.state_data else ([], [], []))
    state_dim = len(ss[0]) if ss else 0
    # Δz 缩放系数（gen_state_data 落盘，存进 config 供推理端保持一致）
    dz_scale = 1.0
    if os.path.exists("state_scale.json"):
        with open("state_scale.json", "r", encoding="utf-8") as f:
            dz_scale = json.load(f).get("dz_scale", 1.0)
    px, py, ps = build_plain_samples(
        args.plain_corpus, stoi, int(args.plain_budget * 1e6), state_dim, args.repeat
    )
    print(f"state samples: {len(sx)} (dim {state_dim}, dz_scale {dz_scale:.1f}), plain samples: {len(px)}")

    xs = torch.tensor(sx + px, dtype=torch.long)
    ys = torch.tensor(sy + py, dtype=torch.long)
    states = torch.tensor(ss + ps, dtype=torch.float32)
    dl = DataLoader(TensorDataset(xs, ys, states), batch_size=args.batch_size, shuffle=True)

    # 3. 模型：BPE 模式用 lm_bpe_big_sft 的 768/16 层配置；char 模式用 large 配置
    if bpe_tok is not None:
        d_model, n_heads, n_layers, block = (ck_cfg["d_model"], ck_cfg["n_heads"],
                                             ck_cfg["n_layers"], ck_cfg["block_size"])
    else:
        d_model, n_heads, n_layers, block = 640, 10, 14, BLOCK - 1
    model = TinyGPT(
        vocab_size=len(stoi),
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, block_size=block,
        state_dim=state_dim, state_token_id=state_token_id,
    )
    # 4. warm start：词表增长时复制旧行
    new_state = model.state_dict()
    n_inherit = 0
    for k, v in old_state.items():
        if k not in new_state:
            continue
        if new_state[k].shape == v.shape:
            new_state[k] = v
            n_inherit += 1
        elif k in ("token_emb.weight", "head.weight"):
            new_state[k][: v.shape[0]] = v  # 词表 +1：复制旧行，新行随机
            n_inherit += 1
            print(f"vocab growth on {k}: {tuple(v.shape)} -> {tuple(new_state[k].shape)}")
    model.load_state_dict(new_state)
    print(f"model: {count_params(model) / 1e6:.1f}M params, inherited {n_inherit} tensors")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bx, by, bs in dl:
            bx, by, bs = bx.to(device), by.to(device), bs.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(bx, by, state=bs)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(bx, by, state=bs)
                loss.backward()
                opt.step()
            tot += loss.item()
            nb += 1
        avg = tot / nb
        print(f"epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if avg < best:
            best = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "config": {
                        "block_size": block, "d_model": d_model, "n_heads": n_heads,
                        "n_layers": n_layers, "state_dim": state_dim,
                        "state_token_id": state_token_id,
                        "dz_scale": dz_scale,
                        "tokenizer": args.bpe or "char",
                    },
                },
                args.out,
            )
    print(f"done. best {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
