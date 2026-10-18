"""统一 tokenizer 抽象层：char-level 与 BPE 无缝切换。

BPE 迁移的目的（昨天诊断的真瓶颈）：char-level 中文压缩率 1.0 字符/token，
BPE 是 1.82——同样的 block_size 能装多 82% 的内容。对语言模型来说，
这直接等于上下文扩容。

设计：
  - CharTokenizer：字符级（旧实现，保持兼容）
  - BPETokenizer：加载 bpe_6400.json（纯 Python BPE，零依赖）
  - 两者接口一致：encode(text) -> ids / decode(ids) -> text / vocab_size

用法：
    from tokenizer import load_tokenizer
    tok = load_tokenizer("bpe_6400.json")   # 或 None = char-level
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe_tokenizer import BPE


class CharTokenizer:
    """字符级 tokenizer（旧默认）：词表 = 语料字符集 + 特殊 token。"""

    def __init__(self, text, specials=("\uE000",)):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        for s in specials:
            if s not in self.stoi:
                self.stoi[s] = len(self.stoi)
        self.itos = {i: c for c, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        inv = {v: k for k, v in self.stoi.items()}
        return "".join(inv.get(i, "") for i in ids)

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def eot_id(self):
        return self.stoi.get("\n", 0)

    def char(self, s):
        """单字符/短串的 id（找不到返回 None）。"""
        return self.stoi.get(s)


class BPETokenizerAdapter:
    """BPE tokenizer 适配器：接口对齐 CharTokenizer。

    特殊 token（[STATE] 的 \uE000）用 BPE 词表之外的独立 id
    （追加到词表末尾），保证和 char 版一样可注入。

    OOV（词表外字符）不会被替换成 id 0（那等于注入假换行）：默认丢弃，
    统计在 oov_total / oov_chars；strict=True 时抛 OOVError。
    """

    def __init__(self, bpe, specials=("\uE000",)):
        self.bpe = bpe
        n_bpe = len(bpe.vocab)
        self.specials = {s: n_bpe + i for i, s in enumerate(specials)}
        self._inv = None

    def encode(self, text, strict=False):
        # 特殊 token 单独处理：文本按特殊 token 切段，BPE 段编码 + 特殊 token id
        out = []
        rest = text
        while self.specials and any(s in rest for s in self.specials):
            for s, sid in self.specials.items():
                if s in rest:
                    head, rest = rest.split(s, 1)
                    out.extend(self.bpe.encode(head, strict=strict))
                    out.append(sid)
                    break
        out.extend(self.bpe.encode(rest, strict=strict))
        return out

    @property
    def oov_total(self):
        return self.bpe.oov_total

    @property
    def oov_chars(self):
        return dict(self.bpe.oov_chars)

    @property
    def oov_rate(self):
        return self.bpe.oov_rate

    def reset_oov_stats(self):
        self.bpe.reset_oov_stats()

    def decode(self, ids):
        if self._inv is None:
            self._inv = {v: k for k, v in self.bpe.vocab.items()}
        out = []
        for i in ids:
            for s, sid in self.specials.items():
                if i == sid:
                    out.append(s)
                    break
            else:
                out.append(self._inv.get(i, ""))
        return "".join(out)

    @property
    def vocab_size(self):
        return len(self.bpe.vocab) + len(self.specials)

    @property
    def eot_id(self):
        return self.bpe.vocab.get("\n", 0)

    def char(self, s):
        return self.bpe.vocab.get(s)


def load_tokenizer(bpe_path=None, fallback_text=""):
    """加载 tokenizer：给了 BPE 路径就用 BPE，否则退回字符级。"""
    if bpe_path and os.path.exists(bpe_path):
        with open(bpe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        bpe = BPE([tuple(p) for p in data["merges"]], data["vocab"])
        return BPETokenizerAdapter(bpe)
    return CharTokenizer(fallback_text) if fallback_text else None
