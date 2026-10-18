"""训练微型语言模型。

流程：
  1. corpus.py 生成语料（问/答对，char-level）
  2. 建 char-level 词表，编码整个语料
  3. 随机切 block_size 窗口做 next-token 预测训练（交叉熵）
  4. 保存最佳权重 + 词表

用法：python train_lm.py --epochs 300
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from corpus import build_corpus
from tinygpt import TinyGPT, count_params

CORPUS_FILE = "corpus.txt"
CKPT_FILE = "lm_checkpoint.pt"


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # 1. 语料
    if not os.path.exists(CORPUS_FILE):
        text, pairs = build_corpus()
        with open(CORPUS_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"corpus built: {len(pairs)} pairs, {len(text)} chars")
    else:
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"corpus loaded: {len(text)} chars")

    # 2. 词表 + 编码
    stoi, itos = build_vocab(text)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    vocab_size = len(stoi)
    print(f"vocab size: {vocab_size}, total tokens: {len(data)}")

    # 3. 训练样本：随机窗口，input = x[:-1], target = x[1:]（两者等长）
    bs = args.block_size
    n_samples = max(1, (len(data) - 1) // bs)
    xs = torch.stack([data[i * bs : i * bs + bs][:-1] for i in range(n_samples)])
    ys = torch.stack([data[i * bs + 1 : i * bs + bs] for i in range(n_samples)])
    print(f"training samples: {len(xs)}")

    dl = DataLoader(TensorDataset(xs, ys), batch_size=args.batch_size, shuffle=True)

    model = TinyGPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        block_size=args.block_size - 1,
    )
    print("model params:", count_params(model))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for bx, by in dl:
            _, loss = model(bx, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        avg = tot / nb
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={avg:.4f}  ({time.time() - t0:.0f}s)")
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "stoi": stoi,
                    "itos": itos,
                    # 存实际使用的 block_size（比 args 小 1，见 xs/ys 切窗）
                    "config": {**vars(args), "block_size": args.block_size - 1},
                },
                CKPT_FILE,
            )
    print(f"done. best loss {best_loss:.4f}, total {time.time() - t0:.0f}s -> {CKPT_FILE}")


if __name__ == "__main__":
    main()
