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


def train_bpe_fast(text, vocab_size=6400, report_every=500):
    """增量版 BPE 训练：pair→位置索引 + 双向链表，复杂度从
    O(n × merges) 降到 O(n + merges × 平均出现数)。

    300 万字符：全扫描版 10 分钟+，增量版几秒——词表训练是基础工具，
    这个优化决定了"在工业语料上训词表"可不可行。
    """
    from collections import defaultdict
    from heapq import heapify, heappop, heappush

    seq = list(text)
    n = len(seq)
    nxt = list(range(1, n + 1))   # 链表：i 的下一个有效下标（n = 结尾）
    prv = list(range(-1, n - 1))  # 链表：i 的上一个有效下标

    pair_pos = defaultdict(set)   # pair -> {positions}
    for i in range(n - 1):
        pair_pos[(seq[i], seq[i + 1])].add(i)

    heap = [(-len(v), p) for p, v in pair_pos.items() if len(v) > 0]
    heapify(heap)

    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    merges = []
    target_merges = max(0, vocab_size - len(vocab))

    def add_pair(a, b, pos):
        pair_pos[(a, b)].add(pos)

    def remove_at(pos, old_pair):
        # pos 处的符号变了：旧 pair 计数减一（位置从集合移除）
        s = pair_pos.get(old_pair)
        if s is not None:
            s.discard(pos)

    while len(merges) < target_merges and heap:
        negf, pair = heappop(heap)
        cnt = len(pair_pos.get(pair, ()))
        if cnt == 0:
            continue
        if cnt != -negf:
            heappush(heap, (-cnt, pair))
            continue
        # 合并 pair 的所有出现（从后往前避免下标失效）
        positions = sorted(pair_pos[pair], reverse=True)
        new_symbol = pair[0] + pair[1]
        for pos in positions:
            # pos 仍有效且内容未变（可能被相邻合并破坏）
            if nxt[pos] == n or seq[nxt[pos]] != pair[1] or seq[pos] != pair[0]:
                continue
            q = nxt[pos]
            # 断开左邻的 pair
            p = prv[pos]
            if p != -1:
                remove_at(p, (seq[p], seq[pos]))
            # 断开右邻的 pair
            r = nxt[q]
            if r != n:
                remove_at(q, (seq[q], seq[r]))
            # 合并：pos 变为 new_symbol，q 标记删除
            seq[pos] = new_symbol
            nxt[pos] = nxt[q]
            if nxt[q] != n:
                prv[nxt[q]] = pos
            seq[q] = None
            # 新 pair 入索引
            if p != -1:
                add_pair(seq[p], new_symbol, p)
            if r != n:
                add_pair(new_symbol, seq[r], pos)
        merges.append(pair)
        # 新 pair 入堆
        for key in [(pair[0], new_symbol), (new_symbol, pair[1])]:
            if len(pair_pos.get(key, ())) > 0:
                heappush(heap, (-len(pair_pos[key]), key))
        if len(merges) % report_every == 0:
            print(f"  merges: {len(merges)}/{target_merges}", flush=True)

    base = len(vocab)
    for i, (a, b) in enumerate(merges):
        vocab[a + b] = base + i
    return merges, vocab


def train_bpe(text, vocab_size=6400):
    """标准字符级 BPE 训练（无预分词）：返回 (merges, vocab)。

    实现：整个文本是一个符号序列，每轮统计最高频相邻符号对并合并。
    注意 words 必须按"序列"而非"字符"计数——第一版写成 Counter(list(text))
    导致键是单个字符、len-1 循环不执行、merges 直接为 0（实测踩过）。
    """
    if len(text) > 500_000:
        return train_bpe_fast(text, vocab_size)
    # 小文本用简单版（代码更直观）
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
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

    for i, (a, b) in enumerate(merges):
        # id 必须是 基础词表大小 + merge 序号。
        # 坑：写成 len(vocab) + i 时 len 每轮在涨，最后一轮 id 会到
        # 2053 + 2×4346 ≈ 10745，远超词表大小——encode 出的 id 全部越界，
        # embedding 层直接 IndexError（实测踩过）。
        vocab[a + b] = len(chars) + i
    return merges, vocab


class BPE:
    """BPE 编解码器（加载 merges/vocab 后使用）。"""

    def __init__(self, merges, vocab):
        self.merges = merges
        self.vocab = vocab
        self.rank = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text, chunk=4000):
        """BPE 编码。长文本分块处理（每块独立编码，块间不跨 merge）——
        全文本单次 encode 是 O(n × merges) 扫描，3000 万字符要几分钟；
        分块后 4000 字符一块，几秒完成（真实 BPE 也是先 pretokenize 分块）。
        """
        if len(text) <= chunk:
            return self._encode_chunk(text)
        out = []
        for i in range(0, len(text), chunk):
            out.extend(self._encode_chunk(text[i : i + chunk]))
        return out

    def _encode_chunk(self, text):
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