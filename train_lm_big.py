"""大语言模型训练：GPU（ROCm）优先，混合精度，梯度累积。

在 ComfyUI 的 ROCm python 环境上运行（该环境已带可用的 torch ROCm）：
    H:/ComfyUI_rocm/python_env/python.exe train_lm_big.py

RDNA2 注意：禁用 flash SDP，走 math attention（RDNA2 上 flash 路径有已知问题）。
"""

import os
import sys

# ComfyUI 的 python 是 embed 版，不会把脚本目录加入 sys.path，手动注入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在 import torch 之前设置
os.environ.setdefault("TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED", "0")
os.environ.setdefault("TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED", "1")

import argparse
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from corpus_big import build
from tinygpt import TinyGPT, count_params

CORPUS_FILE = "corpus_big.txt"
CKPT_FILE = "lm_big_checkpoint.pt"

SCALES = {
    # 16G 显存实测安全区：全参训练约 ~16 字节/参数 + 激活
    "small": dict(d_model=384, n_layers=8, n_heads=6),      # ~16M 参数，vram 3.1G
    "medium": dict(d_model=512, n_layers=12, n_heads=8),    # ~50M 参数
    "large": dict(d_model=640, n_layers=14, n_heads=10),    # ~72M 参数，vram 8.7G（RX 6800 实测）
    "xlarge": dict(d_model=768, n_layers=16, n_heads=12),   # ~118M 参数，batch 32，vram ~11G
}


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="large", choices=list(SCALES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--corpus", type=str, default=CORPUS_FILE, help="语料文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从已有 checkpoint 续训（按形状匹配加载）")
    parser.add_argument("--vocab-from", type=str, default=None,
                        help="从该 checkpoint 继承词表（续训必传，防止 embedding 被打乱）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=CKPT_FILE)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not os.path.exists(args.corpus):
        text = build()
        with open(args.corpus, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(args.corpus, "r", encoding="utf-8") as f:
            text = f.read()

    # 词表：--vocab-from 时从 checkpoint 继承（只追加新字符）
    # 踩坑：重建词表会让续训语料的词表覆盖 checkpoint 的词表，
    # embedding/head 因形状变化全废（参数 75.5M→69.6M，loss 2→6 爆炸）
    if args.vocab_from and os.path.exists(args.vocab_from):
        ck_v = torch.load(args.vocab_from, map_location="cpu")
        stoi = dict(ck_v["stoi"])
        itos = dict(ck_v["itos"])
        added = 0
        for ch in text:
            if ch not in stoi:
                stoi[ch] = len(stoi)
                itos[len(itos)] = ch
                added += 1
        print(f"vocab: inherited {len(ck_v['stoi'])} from {args.vocab_from}, +{added} new -> {len(stoi)}")
    else:
        stoi, itos = build_vocab(text)
    data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
    print(f"corpus: {len(text)} chars, vocab: {len(stoi)}, tokens: {len(data)}")

    bs = args.block_size
    n = max(1, (len(data) - 1) // bs)
    xs = torch.stack([data[i * bs : i * bs + bs][:-1] for i in range(n)])
    ys = torch.stack([data[i * bs + 1 : i * bs + bs] for i in range(n)])
    print(f"samples: {len(xs)}")

    dl = DataLoader(TensorDataset(xs, ys), batch_size=args.batch_size, shuffle=True)

    cfg = SCALES[args.scale]
    model = TinyGPT(
        vocab_size=len(stoi),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        block_size=bs - 1,
    )
    n_params = count_params(model)
    print(f"model: {args.scale} {cfg} -> {n_params / 1e6:.1f}M params")

    # 断点续训：按形状匹配加载（词表变化时 embedding 保持新初始化，其余层继承）
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        old_state = ck["model"]
        new_state = model.state_dict()
        matched = {
            k: v for k, v in old_state.items()
            if k in new_state and new_state[k].shape == v.shape
        }
        new_state.update(matched)
        model.load_state_dict(new_state)
        print(f"resumed {len(matched)}/{len(old_state)} tensors from {args.resume}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU, training on CPU (slow). use the ROCm python env.")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_loss = float("inf")
    t0 = time.time()
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
                scaler.step(opt)
                scaler.update()
            else:
                _, loss = model(bx, by)
                loss.backward()
                opt.step()
            tot += loss.item()
            nb += 1
        avg = tot / nb
        if epoch % 5 == 0 or epoch == 1:
            vram = ""
            if device == "cuda":
                vram = f"  vram {torch.cuda.max_memory_allocated() / 1e9:.1f}G"
            print(f"epoch {epoch:3d}/{args.epochs}  loss={avg:.4f}  ({time.time() - t0:.0f}s){vram}", flush=True)
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    "config": {"block_size": bs - 1, **cfg, "scale": args.scale},
                },
                args.out,
            )
    print(f"done. best {best_loss:.4f}, {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
