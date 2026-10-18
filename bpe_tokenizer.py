"""从零实现 BPE 分词器（参考 MiniMind 的 tokenizer 训练思路，零依赖）。

为什么做：char-level 每字一 token，中文压缩率 1.0 字符/token——
同样的上下文长度要多吃 50-70% 的算力（MiniMind 的 BPE 是 1.5-1.7）。
BPE 把高频字/词合并成一个 token，等于免费扩容上下文。

实现（标准 BPE，简化版）：
  1. 基础词表 = 语料中出现的所有字符
  2. 反复统计最高频的相邻符号对，合并成一个新符号
  3. 达到目标词表大小时停止
  4. 编码时按 merges 顺序贪心合并

用法：
    python bpe_tokenizer.py --train --vocab-size 6400 --out bpe_6400.json
    python bpe_tokenizer.py --bench      # 测中文压缩率
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SPECIALS = ["\n", " ", "：", "？", "！", "，", "。"]


def load_corpus(paths):
    """合并多个语料文件（jsonl 会自动抽出 text/conversations 字段）。"""
    parts = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            if p.endswith(".jsonl"):
                for line in f:
                    obj = json.loads(line)
                    if "conversations" in obj:
                        parts.extend(m["content"] for m in obj["conversations"])
                    elif "text" in obj:
                        parts.append(obj["text"])
            else:
                parts.append(f.read())
    return "".join(parts)


def train_bpe(text, vocab_size=6400):
    """标准字符级 BPE 训练（无预分词）：返回 (merges, vocab)。

    实现：整个文本是一个符号序列，每轮统计最高频相邻符号对并合并。
    注意 words 必须按"序列"而非"字符"计数——第一版写成 Counter(list(text))
    导致键是单个字符、len-1 循环不执行、merges 直接为 0（实测踩过）。
    """
    # 基础词表：字符级
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    print(f"base chars: {len(vocab)}")

    seq = list(text)  # 符号序列（每个元素是一个"符号"，初始是单字符）
    merges = []
    target_merges = max(0, vocab_size - len(vocab))
    while len(merges) < target_merges:
        pair_freq = Counter(zip(seq[:-1], seq[1:]))
        if not pair_freq:
            break
        best = max(pair_freq, key=pair_freq.get)
        # 一次性替换所有不重叠的 best 出现
        out, i, n = [], 0, len(seq)
        while i < n:
            if i < n - 1 and seq[i] == best[0] and seq[i + 1] == best[1]:
                out.append(seq[i] + seq[i + 1])
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
        merges.append(best)
        if len(merges) % 200 == 0:
            print(f"  merges: {len(merges)}/{target_merges}, "
                  f"seq len: {len(seq)} (压缩 {(1 - len(seq) / len(text)) * 100:.1f}%)",
                  flush=True)

    base = len(vocab)
    for i, (a, b) in enumerate(merges):
        # id 必须是 基础词表大小 + merge 序号。
        # 坑：写成 len(vocab) + i 时 len 每轮在涨，最后一轮 id 会到
        # 2053 + 2×4346 ≈ 10745，远超词表大小——encode 出的 id 全部越界，
        # embedding 层直接 IndexError（实测踩过）。
        vocab[a + b] = base + i
    return merges, vocab


class BPE:
    """BPE 编解码器（加载 merges/vocab 后使用）。"""

    def __init__(self, merges, vocab):
        self.merges = merges
        self.vocab = vocab
        self.rank = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text):
        symbols = list(text)
        while True:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self.rank and (best_rank is None or self.rank[pair] < best_rank):
                    best, best_rank = pair, self.rank[pair]
            if best is None:
                break
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(best[0] + best[1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return [self.vocab.get(s, 0) for s in symbols]

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, "") for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=6400)
    parser.add_argument("--out", type=str, default="bpe_6400.json")
    parser.add_argument("--corpus", nargs="*", default=[
        "corpus_mixed.txt", "corpus_dialog.txt", "sft_data_big.jsonl"])
    args = parser.parse_args()

    if args.train:
        text = load_corpus(args.corpus)
        print(f"corpus: {len(text)} chars")
        merges, vocab = train_bpe(text, args.vocab_size)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"merges": [list(p) for p in merges], "vocab": vocab},
                      f, ensure_ascii=False)
        print(f"trained: vocab {len(vocab)} + merges {len(merges)} -> {args.out}")

    if args.bench and os.path.exists(args.out):
        with open(args.out, "r", encoding="utf-8") as f:
            data = json.load(f)
        bpe = BPE([tuple(p) for p in data["merges"]], data["vocab"])
        samples = [
            "世界模型是一类预测环境动态的神经网络模型。",
            "你好，我是小WM，一个会看会说会走的迷你 agent。",
            "长城是古代防御工程体系，秦把战国长城连接。",
        ]
        total_chars = total_tokens = 0
        for s in samples:
            ids = bpe.encode(s)
            total_chars += len(s)
            total_tokens += len(ids)
            print(f"{len(s):3d} 字 -> {len(ids):3d} token | {s[:20]}...")
        print(f"\n压缩率: {total_chars / max(1, total_tokens):.2f} 字符/token "
              f"(char-level 是 1.00；MiniMind 的 BPE 是 1.5-1.7)")
        # 抽样看看学到的词
        inv = {v: k for k, v in data["vocab"].items() if len(k) >= 2}
        learned = [k for k in inv.values()][:20]
        print(f"学到的多字符 token 示例: {learned}")


if __name__ == "__main__":
    main()