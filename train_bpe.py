"""BPE 版预训练（修架构第一步：char-level → BPE 全链路迁移）。

配置参考 MiniMind 的 minimind2-small：26M 参数、8 层、d_model=512、词表 6400。
我们的语料比 MiniMind 小 100 倍（20 万 vs 2000 万字符），所以：
  - block_size 256 token（BPE 下约 460 字符内容，比 char-level 的 96 字符多 4.8 倍）
  - epochs 15（小语料多跑几轮）
  - 数据 = 手写全集（对话 + 知识 + 世界叙述 + SFT QA）

用法：python train_bpe.py --out lm_bpe_pretrain.pt
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import load_tokenizer
from tinygpt import TinyGPT, count_params

BLOCK = 256  # BPE token 上下文（≈460 中文字符）


class TokenDataset(Dataset):
    """把文本切成 block 长度的 token 序列。"""

    def __init__(self, text, tok, block):
        ids = tok.encode(text)
        # 防御：BPE 偶发越界（词表/编码不一致）时 clip，别让 embedding 崩。
        # 根因排查中（default venv 复现零越界，ComfyUI python 训练时报错），
        # 先保证可跑，同时打印诊断。
        n_over = sum(1 for i in ids if i >= tok.vocab_size)
        if n_over:
            print(f"[warn] {n_over} 个越界 token 已 clip（max id {max(ids)}, "
                  f"vocab {tok.vocab_size}）")
            ids = [min(i, tok.vocab_size - 1) for i in ids]
        n = max(1, (len(ids) - 1) // block)
        self.xs = [ids[i * block : i * block + block][:-1] for i in range(n)]
        self.ys = [ids[i * block + 1 : i * block + block] for i in range(n)]
        self.n_tokens = len(ids)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self.ys[i]


def collate(batch, block, pad_id):
    xs, ys = [], []
    for x, y in batch:
        px, py = block - len(x), block - len(y)
        xs.append(x + [pad_id] * px)
        ys.append(y + [pad_id] * py)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def load_corpus_text(paths):
    parts = []
    for p in paths:
        if not os.path.exists(p):
            continue
        if p.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    if "conversations" in obj:
                        parts.extend(m["content"] for m in obj["conversations"])
                    elif "text" in obj:
                        parts.append(obj["text"])
        else:
            with open(p, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="*", default=[
        "corpus_mixed.txt", "corpus_dialog.txt", "sft_data_big.jsonl"])
    parser.add_argument("--bpe", type=str, default="bpe_6400.json")
    parser.add_argument("--out", type=str, default="lm_bpe_pretrain.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(args.bpe)
    if tok is None:
        sys.exit("BPE 词表不存在，先跑 python bpe_tokenizer.py --train")
    text = load_corpus_text(args.corpus)
    print(f"corpus: {len(text)} chars")

    ds = TokenDataset(text, tok, BLOCK)
    print(f"tokens: {ds.n_tokens} (压缩率 {len(text) / ds.n_tokens:.2f} 字符/token), "
          f"samples: {len(ds)}, block: {BLOCK} token")

    pad_id = tok.eot_id
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, BLOCK - 1, pad_id))  # 与 block_size 对齐

    model = TinyGPT(vocab_size=tok.vocab_size, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers,
                    block_size=BLOCK - 1, state_dim=0)
    print(f"model: {count_params(model) / 1e6:.1f}M params, device: {device}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best, t0 = float("inf"), time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            if scaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(bx, by)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(bx, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()
            nb += 1
        avg = tot / max(1, nb)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}  ({time.time() - t0:.0f}s)",
              flush=True)
        if avg < best:
            best = avg
            torch.save({
                "model": model.state_dict(),
                "stoi": {k: v for k, v in tok.bpe.vocab.items()} if hasattr(tok, "bpe")
                         else tok.stoi,
                "itos": None,
                "config": {
                    "block_size": BLOCK - 1, "d_model": args.d_model,
                    "n_heads": args.n_heads, "n_layers": args.n_layers,
                    "state_dim": 0, "state_token_id": None,
                    "tokenizer": "bpe_6400.json",
                },
            }, args.out)
    print(f"done. best {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
