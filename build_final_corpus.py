"""构建最终训练语料 v2：多分片扩充版（目标 5000 万字符）。

相对 v1 的改进：
  - SkyPile 用 3 个分片（各 Range 40MB），文本量 3 倍
  - zhwiki 用 2 个分片，百科文本 2 倍，定义句 QA 2 倍（12 万对）
  - QA 数据更多，占比保持 ~12%
  - 词表过滤阈值 50（实测 4997 词表，仅损失 0.34% 文本）

输出 corpus_final_50m.txt。
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus_big import build as build_qa
from download_corpus import CACHE, extract_definitions, parse_skypile, parse_zhwiki

SKYPILE_FILES = [f"{CACHE}/skypile.jsonl", f"{CACHE}/skypile_2.jsonl", f"{CACHE}/skypile_3.jsonl"]
ZHWIKI_FILES = [f"{CACHE}/zhwiki.parquet", f"{CACHE}/zhwiki_2.parquet"]


def build(skypile_budget, zhwiki_budget, min_char_freq):
    # 1. SkyPile 工业文本（3 分片轮流取，保证多样性）
    texts = []
    n = 0
    for path in SKYPILE_FILES:
        if not os.path.exists(path):
            continue
        for t in parse_skypile(path):
            texts.append(t)
            n += len(t)
            if n >= skypile_budget:
                break
        if n >= skypile_budget:
            break
    skypile_chars = sum(len(t) for t in texts)
    print(f"skypile: {len(texts)} docs, {skypile_chars / 1e6:.1f}M chars")

    # 2. zhwiki 百科文本（2 分片）
    n = 0
    for path in ZHWIKI_FILES:
        if not os.path.exists(path):
            continue
        for t in parse_zhwiki(path):
            texts.append(t)
            n += len(t)
            if n >= zhwiki_budget:
                break
        if n >= zhwiki_budget:
            break
    zhwiki_chars = sum(len(t) for t in texts) - skypile_chars
    print(f"zhwiki: {zhwiki_chars / 1e6:.1f}M chars")

    # 3. 定义句 -> 高质量 QA（两个分片都抽）
    qa_lines = []
    for path in ZHWIKI_FILES:
        if not os.path.exists(path):
            continue
        for d in extract_definitions(path):
            if "：" not in d:
                continue
            title, defn = d.split("：", 1)
            title = title.strip()
            defn = defn.strip()
            if not title or not defn or len(title) > 30:
                continue
            qa_lines.append(f"问：{title}是什么？\n答：{defn}\n")
            qa_lines.append(f"问：什么是{title}？\n答：{defn}\n")
    print(f"definition QA: {len(qa_lines) // 2} pairs")

    # 4. 手写 QA
    handwritten = [line + "\n" for line in build_qa().split("\n") if line.strip()]
    qa_lines.extend(handwritten)

    # 5. 混合
    corpus = "\n".join(texts) + "\n" + "\n".join(qa_lines) + "\n"

    # 6. 词表过滤
    cnt = Counter(corpus)
    keep = {c for c, n in cnt.items() if n >= min_char_freq}
    removed = sum(n for c, n in cnt.items() if c not in keep)
    corpus = "".join(c for c in corpus if c in keep)
    print(f"vocab: {len(cnt)} -> {len(keep)} (removed {removed / len(corpus) * 100:.2f}% of text)")
    return corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skypile", type=float, default=22.0, help="SkyPile 字符预算（百万）")
    parser.add_argument("--zhwiki", type=float, default=12.0, help="zhwiki 字符预算（百万）")
    parser.add_argument("--min-freq", type=int, default=50)
    parser.add_argument("--vocab-from", type=str, default=None,
                        help="锁定词表：只保留该 checkpoint stoi 中的字符（续训时防止词表漂移）")
    parser.add_argument("--out", type=str, default="corpus_v3.txt")
    args = parser.parse_args()

    corpus = build(
        args.skypile * 1e6,
        args.zhwiki * 1e6,
        args.min_freq,
    )

    # 词表锁定：只保留指定 checkpoint 词表中的字符
    if args.vocab_from:
        import torch

        ck = torch.load(args.vocab_from, map_location="cpu")
        allowed = set(ck["stoi"].keys())
        before = len(corpus)
        corpus = "".join(c for c in corpus if c in allowed)
        print(f"vocab locked to {len(allowed)} chars (from {args.vocab_from}), "
              f"removed {(before - len(corpus)) / before * 100:.2f}% of text")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"\n{args.out}: {len(corpus) / 1e6:.1f}M chars")
