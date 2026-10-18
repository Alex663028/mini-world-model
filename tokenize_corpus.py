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
    np.save(args.out, arr)
    print(f"done: {len(ids)} tokens ({len(text) / len(ids):.2f} 字符/token), "
          f"{time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
