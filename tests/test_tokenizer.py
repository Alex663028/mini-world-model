"""分词器测试：编解码往返、特殊 token、词表边界、OOV 处理。

重点覆盖真实踩过的坑：
1. BPE 词表 id 分配错误导致 encode 出的 id 超过 embedding 行数
   （最后一轮冲到 10745，词表只有 6400）。
2. OOV 字符被静默编码成 id 0（= "\\n"），往训练数据里注入假换行。
"""

import os
import subprocess
import sys

import pytest

from bpe_tokenizer import BPE, OOVError, train_bpe, train_bpe_fast
from tokenizer import CharTokenizer, load_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BPE_PATH = os.path.join(ROOT, "bpe_6400.json")

SAMPLE = "你好，我是小WM。世界模型是我的眼睛，语言模型是我的嘴。"


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer(SAMPLE * 3)
    ids = tok.encode(SAMPLE)
    assert isinstance(ids, list)
    assert tok.decode(ids) == SAMPLE, "char 模式必须无损往返"


def test_char_special_token():
    tok = CharTokenizer(SAMPLE)
    assert tok.char("\uE000") == tok.vocab_size - 1, "特殊 token 应追加在末尾"
    assert tok.vocab_size > len(set(SAMPLE))


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_roundtrip():
    tok = load_tokenizer(BPE_PATH)
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE, "BPE 必须无损往返"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_ids_in_range():
    """所有 encode 出的 id 必须 < embedding 行数。

    踩过的坑：vocab[a+b] = len(vocab) + i 让 len 每轮增长，
    最后一轮 id 到 10745，训练时 embedding 直接 IndexError。
    """
    tok = load_tokenizer(BPE_PATH)
    text = SAMPLE * 50 + "世界模型语言模型目标规划画面长城光速秦始皇"
    for i in tok.encode(text):
        assert 0 <= i < tok.vocab_size, f"id {i} 超出词表 {tok.vocab_size}"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_state_token():
    """[STATE] 必须映射到独立 id（在 BPE 词表之外）。"""
    tok = load_tokenizer(BPE_PATH)
    ids = tok.encode(f"问：你看到了什么？\n\uE000\n答：")
    assert tok.specials["\uE000"] in ids, "STATE token 必须出现在 ids 里"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_compresses_better_than_char():
    """BPE 的意义：同等文本 token 数更少。

    这里用一句通用中文（不含训练语料里的高频术语），压缩率约 1.2-1.3；
    README 里的 2.22 是手写对话语料（术语反复出现，BPE 学到长片段）。
    断言只验证"确实更省"，不绑定具体数值。
    """
    bpe = load_tokenizer(BPE_PATH)
    char = CharTokenizer(SAMPLE * 3)
    n_bpe = len(bpe.encode(SAMPLE))
    n_char = len(char.encode(SAMPLE))
    assert n_bpe < n_char, f"BPE({n_bpe}) 应少于 char({n_char})"
    assert n_char / n_bpe > 1.15, f"压缩率偏低：{n_char / n_bpe:.2f}"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_compresses_common_words():
    """当前词表训练于工业语料：常用双字词是单 token，压缩率约 1.2-1.4。

    注意：词表是**域相关**的。这个 6400 词表用 300 万字符工业语料训练，
    "中国/发展/公司/经济"是单 token；而"世界模型/语言模型"这类项目术语
    没有合成长片段（早期手写对话语料训的词表才有，压缩率能到 2.22）。
    """
    bpe = load_tokenizer(BPE_PATH)
    text = "中国经济发展公司的社会问题研究工作生活技术国家"
    n_bpe = len(bpe.encode(text))
    assert len(text) / n_bpe > 1.3, f"工业常用词压缩率应 >1.3，实际 {len(text) / n_bpe:.2f}"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_vocab_size():
    tok = load_tokenizer(BPE_PATH)
    assert tok.vocab_size == 6401, "6400 BPE + 1 个 STATE token"


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_decode_unknown_id():
    """未知 id 不应崩溃（解码为跳过的空串）。"""
    tok = load_tokenizer(BPE_PATH)
    out = tok.decode([0, 1, 999999])
    assert isinstance(out, str)


# ---------- OOV：不得静默替换成 id 0 ----------


def test_bpe_oov_not_replaced_by_zero():
    """词表外字符必须被丢弃，不能映射成 id 0。

    回归用例：旧实现是 ``self.vocab.get(s, 0)``，词表里 id 0 是 "\\n"，
    于是 encode("a😀b") 得到 [0, 0, 1]，decode 回来是 "a\\nb"——
    训练数据里凭空多了换行，而且毫无提示。
    """
    bpe = BPE(merges=[], vocab={"a": 0, "b": 1, "\n": 2})
    ids = bpe.encode("a😀b")
    assert ids == [0, 1], f"OOV 应被丢弃，实际 {ids}"
    assert bpe.decode(ids) == "ab"
    assert bpe.oov_total == 1
    assert "😀" in bpe.oov_chars
    # 合法字符的 id 不受影响
    assert bpe.encode("\n") == [2]


def test_bpe_oov_strict_raises():
    """严格模式：遇到 OOV 直接抛错，方便训练前把语料问题挡在门口。"""
    bpe = BPE(merges=[], vocab={"a": 0})
    with pytest.raises(OOVError):
        bpe.encode("a😀", strict=True)
    # 非严格模式只计数，不抛
    assert bpe.encode("a😀") == [0]


def test_bpe_oov_unk_id_optional():
    """预留了 UNK 位时映射到 UNK（而不是 id 0）。"""
    bpe = BPE(merges=[], vocab={"a": 0}, unk_id=99)
    assert bpe.encode("a😀") == [0, 99]


@pytest.mark.skipif(not os.path.exists(BPE_PATH), reason="bpe_6400.json 不存在")
def test_bpe_real_vocab_oov_is_not_newline():
    """真实词表 + emoji/生僻字：decode 必须等于丢掉 OOV 后的文本。"""
    tok = load_tokenizer(BPE_PATH)
    tok.reset_oov_stats()
    ids = tok.encode("你好😀世界𠮷")
    assert 0 not in ids, "id 0 是换行，绝不能由 OOV 产生"
    assert tok.decode(ids) == "你好世界"
    assert tok.oov_total == 2, f"应统计到 2 个 OOV，实际 {tok.oov_total}"


# ---------- 空语料：必须报错，不能静默产出空词表 ----------


def test_train_bpe_empty_corpus_raises():
    with pytest.raises(ValueError):
        train_bpe("")
    with pytest.raises(ValueError):
        train_bpe_fast("")


def test_train_cli_rejects_empty_corpus(tmp_path):
    """--train 在语料缺失/过短时必须非零退出，且不写出词表文件。"""
    out = tmp_path / "out.json"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "bpe_tokenizer.py"),
         "--train", "--corpus", str(tmp_path / "definitely_missing.txt"),
         "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        timeout=120,
    )
    assert proc.returncode != 0, "空语料必须非零退出"
    assert not out.exists(), "不能在失败时写出空词表"
