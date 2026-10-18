"""MiniMind 式 SFT 训练器（参考 jingyaogong/minimind 的 trainer 设计）。

相比"预训练式续训"的关键改进：
  1. **loss mask**：只对 assistant 部分计算 loss（省算力、聚焦回答）。
  2. **cosine 学习率 + warmup**：固定 lr 在收敛末期浪费步数。
  3. **梯度裁剪**：小模型训练稳定性。
  4. **Tool Call 轨迹**：assistant 输出工具调用，tool 返回结果，assistant 总结——
     教模型自己决定调哪个工具（agent 化的关键一步）。
  5. **断点续训含优化器状态**（之前只存模型权重，优化器状态全丢）。

关键约定（踩过的坑）：TinyGPT 的 labels 是 ids **右移一位**（预测下一 token）。
SFT 的 mask 必须在这个右移之后施加——若让 labels 与 ids 平行，
等于教模型"预测自己"，loss 会假性趋零（实测 0.0012 但生成全是"："）。

用法：
    python train_sft.py --data sft_data.jsonl --resume lm_final2.pt --out lm_sft.pt
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tinygpt import TinyGPT, count_params
from tokenizer import load_tokenizer

IGNORE = -100  # loss mask 的忽略值


class SFTDataset(Dataset):
    """conversations JSONL → (ids, learn_flags)，只有 assistant 段计入 loss。

    BPE 模式：tok 非空时用 tok.encode（子词编码），否则字符级。
    """

    def __init__(self, path, stoi, block_size, tok=None):
        self.tok = tok
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ids, learn = self._encode(obj["conversations"], stoi, block_size)
                if sum(learn) >= 2:
                    self.samples.append((ids, learn))

    def _enc(self, s, stoi):
        if self.tok is not None:
            return self.tok.encode(s)
        return [stoi[c] for c in s if c in stoi]

    def _encode(self, conv, stoi, block):
        ids, learn = [], []

        def add(piece, is_target):
            toks = self._enc(piece, stoi)
            ids.extend(toks)
            learn.extend([is_target] * len(toks))

        for m in conv:
            role = m["role"]
            if role == "system":
                add("系统：" + m["content"] + "\n", False)
            elif role == "user":
                add("用户：" + m["content"] + "\n", False)
            elif role == "tool":
                add("结果：" + m["content"] + "\n", False)
            elif role == "assistant":
                add("助手：", False)  # 角色前缀不学
                content = m.get("content") or ""
                if m.get("tool_calls"):
                    calls = m["tool_calls"]
                    if isinstance(calls, str):
                        calls = json.loads(calls)
                    # 带参数的工具调用（两项目融合点）：
                    # [调用工具 world_act, direction=西北]
                    parts = []
                    for c in calls:
                        args = c.get("arguments") or {}
                        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
                        parts.append(c.get("name", "") + (f", {arg_str}" if arg_str else ""))
                    content = "[调用工具 " + "，".join(parts) + "]" + content
                add(content, True)  # 只有回答内容学
                add("\n", True)
            if len(ids) >= block:
                break
        return ids[:block], learn[:block]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch, block_size, pad_id):
    """右移 + loss mask：labels[i] = ids[i+1]，非 assistant 段置 IGNORE。"""
    xs, ys = [], []
    for ids, learn in batch:
        ids = list(ids)
        learn = list(learn)
        while len(ids) < block_size:
            ids.append(pad_id)
            learn.append(False)
        xs.append(ids[:-1])
        ys.append([ids[i + 1] if learn[i + 1] else IGNORE
                   for i in range(block_size - 1)])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


def cosine_lr(step, total, warmup, base_lr, min_ratio=0.1):
    """cosine 衰减 + 线性 warmup（MiniMind 风格）。"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (min_ratio + (1 - min_ratio) * cos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sft_data.jsonl")
    parser.add_argument("--resume", type=str, default="lm_final2.pt")
    parser.add_argument("--out", type=str, default="lm_sft.pt")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--bpe", type=str, default="bpe_6400.json",
                        help="BPE 词表路径；空串 = 字符级（旧行为）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer：BPE（新）或字符级（旧）。BPE 下 tokenizer 自带词表，
    # resume 的 stoi 只用于 warm start 的 shape 匹配。
    tok = load_tokenizer(args.bpe) if args.bpe else None
    stoi = tok.bpe.vocab if tok is not None else None

    ck = torch.load(args.resume, map_location="cpu")
    ck_stoi, ck_itos, state = ck["stoi"], ck["itos"], ck["model"]
    base_cfg = ck["config"]

    # SFTDataset 用 tokenizer 接口（BPE 或 char 都走 encode）
    class _Enc:
        def __init__(self, tok, stoi):
            self.tok = tok
            self.stoi = stoi
        def __getitem__(self, s):
            if self.tok is not None:
                return None  # BPE 路径不走 dict
            return self.stoi.get(s, None)
        def get(self, s, default=None):
            return self.stoi.get(s, default) if self.stoi else default

    enc = _Enc(tok, ck_stoi)
    ds = SFTDataset(args.data, enc, args.block_size, tok=tok)
    n_learn = sum(sum(l) for _, l in ds.samples)
    n_total = sum(len(i) for i, _ in ds.samples)
    print(f"sft samples: {len(ds)}, learnable tokens: {n_learn}/{n_total} "
          f"({n_learn / max(1, n_total) * 100:.1f}% —— 续训模式下这部分是 100%)")

    pad_id = tok.eot_id if tok is not None else (ck_stoi.get(" ", ck_stoi.get("\n", 0)))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, args.block_size - 1, pad_id))

    vocab_n = tok.vocab_size if tok is not None else len(ck_stoi)
    model = TinyGPT(vocab_size=vocab_n, d_model=base_cfg["d_model"],
                    n_heads=base_cfg["n_heads"], n_layers=base_cfg["n_layers"],
                    block_size=args.block_size - 1,
                    state_dim=base_cfg.get("state_dim", 0),
                    state_token_id=base_cfg.get("state_token_id"))
    new_state = model.state_dict()
    inherited = 0
    for k, v in state.items():
        if k not in new_state:
            continue
        if new_state[k].shape == v.shape:
            new_state[k] = v
            inherited += 1
        elif k in ("token_emb.weight", "head.weight"):
            new_state[k][: v.shape[0]] = v
            inherited += 1
    model.load_state_dict(new_state)
    model = model.to(device)
    print(f"model: {count_params(model) / 1e6:.1f}M, inherited {inherited} tensors")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = args.epochs * len(dl)
    warmup = int(total_steps * args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best, step, t0, lr = float("inf"), 0, time.time(), args.lr
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bx, by in dl:
            lr = cosine_lr(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(bx, by)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(bx, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            tot += loss.item()
            nb += 1
            step += 1
        avg = tot / max(1, nb)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}  lr={lr:.2e}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if avg < best:
            best = avg
            torch.save({
                "model": model.state_dict(),
                "stoi": tok.bpe.vocab if tok is not None else ck_stoi,
                "itos": None,   # BPE 模式 decode 靠 bpe.vocab（config 里记 tokenizer 路径）
                "opt": opt.state_dict(),       # MiniMind 式：优化器状态也存
                "config": {**base_cfg, "block_size": args.block_size - 1,
                           "tokenizer": args.bpe or "char"},
            }, args.out)
    print(f"done. best {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()