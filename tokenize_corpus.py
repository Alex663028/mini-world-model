"""语料预编码：BPE encode 后缓存 token ids（.npy），训练直接读。

为什么：BPE encode 是 O(n × merges) 的扫描，3000 万字符一次要几分钟。
但只需跑一次——缓存后 train_bpe.py 直接读，省掉每次训练的编码开销。

用法：python tokenize_corpus.py --corpus corpus_v3.txt --out tokens_v3.npy
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="corpus_v3.txt")
    parser.add_argument("--bpe", type=str, default="bpe_6400.json")
    parser.add_argument("--out", type=str, default="tokens_v3.npy")
    parser.add_argument("--max-chars", type=int, default=25_000_000,
                        help="最多编码多少字符（控制训练规模）")
    args = parser.parse_args()

    tok = load_tokenizer(args.bpe)
    if tok is None:
        sys.exit("BPE 词表不存在")

    text = open(args.corpus, "r", encoding="utf-8").read(args.max_chars)
    print(f"corpus: {len(text)} chars -> encoding...", flush=True)
    t0 = time.time()
    ids = []
    # 按 50 万字符分批编码 + 进度（内存友好）
    step = 500_000
    for i in range(0, len(text), step):
        ids.extend(tok.encode(text[i : i + step]))
        if (i // step) % 4 == 0:
            print(f"  {i / len(text) * 100:.0f}% ({time.time() - t0:.0f}s)", flush=True)
    arr = np.array(ids, dtype=np.int32)
    if arr.size == 0:
        sys.exit("[tokenize] 编码结果为空：语料为空，或所有字符都在 BPE 词表之外")
    np.save(args.out, arr)
    print(f"done: {len(ids)} tokens ({len(text) / len(ids):.2f} 字符/token), "
          f"{time.time() - t0:.0f}s -> {args.out}")
    # OOV 丢弃率必须显式报出来：词表外字符现在被丢弃（以前会静默变成 id 0 =
    # "\n"，污染训练数据），丢弃率高说明语料/词表不匹配，得先处理
    if getattr(tok, "oov_total", 0):
        top = sorted(tok.oov_chars.items(), key=lambda kv: -kv[1])[:10]
        shown = "、".join(f"{c!a}×{n}" for c, n in top)
        print(f"[warn] OOV 丢弃 {tok.oov_total} 字符（{tok.oov_rate * 100:.4f}%），"
              f"高频：{shown}")
    else:
        print("OOV: 0（语料全部落在词表覆盖范围内）")


if __name__ == "__main__":
    main()
