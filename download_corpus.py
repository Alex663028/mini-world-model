"""下载工业级中文预训练语料并清洗混合。

数据源（均为公开数据集，经过官方清洗/去重/隐私过滤）：
  - SkyPile-150B（天工）：中文网页语料，取前 20MB（jsonl 行式，Range 截断无碍）
  - BAAI CCI3-HQ（智源）：中文高质量语料，取前 20MB
  - wikimedia/wikipedia 20231101.zh：中文维基百科，下载 1 个完整 parquet 分片
    并从中抽取"定义句"作为常识三元组的文本化替代（Snowman 无公开下载源）

输出 corpus_industrial.txt（目标 1500-2500 万字符，控制训练时间）。

用法：python download_corpus.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HF = "https://hf-mirror.com"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
CACHE = "corpus_cache"
SKYPILE_URL = f"{HF}/datasets/Skywork/SkyPile-150B/resolve/main/data/2020-40_zh_head_0000.jsonl"
ZHWIKI_URL = f"{HF}/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-00000-of-00006.parquet"

TARGET_CHARS = 20_000_000  # 混合后目标字符数（char-level 训练时间可控）


def _download(url, out, nbytes=None):
    """下载并落到 out。

    优先 curl（走系统代理 + CDN，hf-mirror 上比 urllib 稳得多）；机器上没有
    curl（部分精简 Windows / 容器）时回退 urllib，别让整条语料链路挂掉。
    """
    cmd = ["curl", "-sL", "--retry", "3", "--retry-delay", "3", "-A", UA]
    if nbytes:
        cmd += ["-r", f"0-{nbytes - 1}"]
    cmd += ["-o", out, url]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[download] 未找到 curl，回退 urllib ...", flush=True)
        headers = {"User-Agent": UA}
        if nbytes:
            headers["Range"] = f"bytes=0-{nbytes - 1}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp, open(out, "wb") as f:
            f.write(resp.read())
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        raise RuntimeError(f"下载失败或为空文件：{url} -> {out}")
    print(f"fetched {os.path.getsize(out) / 1e6:.1f}MB -> {out}")


def fetch_range(url, nbytes, out):
    """Range 截取前 nbytes 字节（curl 走代理+CDN，urllib 会被 hf-mirror 限流）。"""
    _download(url, out, nbytes=nbytes)


def fetch_full(url, out):
    _download(url, out)


def cjk_ratio(t):
    if not t:
        return 0.0
    cjk = sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    return cjk / len(t)


def clean_text(t):
    """通用清洗：去 HTML 标签、压缩空白、过滤非中文为主的文本。"""
    if not t:
        return None
    t = re.sub(r"<[^>]{1,200}>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 60 or len(t) > 2000:
        return None
    if cjk_ratio(t) < 0.55:
        return None
    # 过滤明显垃圾（连续符号/外文占比过高）
    if len(re.findall(r"[A-Za-z]", t)) / len(t) > 0.5:
        return None
    return t


def dedup_append(texts, seen, out_f):
    kept = 0
    for t in texts:
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out_f.write(t + "\n")
        kept += 1
    return kept


def parse_skypile(path):
    """SkyPile jsonl：{"text": "..."}"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = clean_text(obj.get("text", ""))
            if t:
                yield t


def parse_zhwiki(path):
    """wikimedia parquet：id / url / title / text 四列。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.column_names
    text_col = "text" if "text" in cols else cols[-1]
    for chunk in table.column(text_col).to_pylist():
        t = clean_text(chunk)
        if t:
            yield t


def extract_definitions(path):
    """从 zhwiki 抽取定义句（"X 是……"句式）作为常识三元组文本化。

    这是 Snowman（无公开下载源）的替代：维基首句定义本身就是高质量三元组文本。
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    titles = table.column("title").to_pylist() if "title" in table.column_names else None
    texts = table.column("text").to_pylist() if "text" in table.column_names else None
    if not titles or not texts:
        return
    n = 0
    for title, text in zip(titles, texts):
        if not title or not text:
            continue
        first = re.split(r"[。！？\n]", text.strip(), maxsplit=1)[0]
        if not first or len(first) < 15 or len(first) > 120:
            continue
        if cjk_ratio(first) < 0.6:
            continue
        if not re.match(r"^[\u4e00-\u9fffA-Za-z0-9·]+", title):
            continue
        yield f"{title}：{first}。"
        n += 1
        if n >= 60000:  # 定义句上限
            break


def main():
    os.makedirs(CACHE, exist_ok=True)
    seen = set()

    # ---- 1. 下载 ----
    # 注意：BAAI/CCI3-HQ 是授权受限数据集（gated），无公开下载源，已弃用。
    # 主力配比：SkyPile 55% / zhwiki 35% / 定义句 10% / 手写问答 5%（CCI 的份额由 SkyPile 补足）
    skypile_raw = f"{CACHE}/skypile.jsonl"
    zhwiki_raw = f"{CACHE}/zhwiki.parquet"
    if not os.path.exists(skypile_raw):
        fetch_range(SKYPILE_URL, 40 * 1024 * 1024, skypile_raw)
        time.sleep(5)
    if not os.path.exists(zhwiki_raw):
        fetch_full(ZHWIKI_URL, zhwiki_raw)

    # ---- 2. 清洗混合 ----
    out_path = "corpus_industrial.txt"
    budget = {
        "skypile": int(TARGET_CHARS * 0.55),
        "zhwiki": int(TARGET_CHARS * 0.35),
        "defs": int(TARGET_CHARS * 0.10),
        "qa": int(TARGET_CHARS * 0.05),
    }
    stats = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for name, parser in [("skypile", parse_skypile), ("zhwiki", parse_zhwiki)]:
            raw = {"skypile": skypile_raw, "zhwiki": zhwiki_raw}[name]
            if not os.path.exists(raw):
                stats[name] = 0
                continue
            n_chars, n_docs = 0, 0
            for t in parser(raw):
                out.write(t + "\n")
                n_chars += len(t)
                n_docs += 1
                if n_chars >= budget[name]:
                    break
            stats[name] = (n_docs, n_chars)
            print(f"{name}: {n_docs} docs, {n_chars / 1e6:.1f}M chars")

        # 定义句（常识三元组文本化）
        n_chars, n_docs = 0, 0
        for t in extract_definitions(zhwiki_raw):
            out.write(t + "\n")
            n_chars += len(t)
            n_docs += 1
            if n_chars >= budget["defs"]:
                break
        stats["defs"] = (n_docs, n_chars)
        print(f"defs: {n_docs} definition sentences, {n_chars / 1e6:.1f}M chars")

        # 手写问答对（保证模型保留问答格式能力）
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from corpus_big import build as build_qa

        qa_text = build_qa()
        n_chars = 0
        for line in qa_text.split("\n"):
            out.write(line + "\n")
            n_chars += len(line) + 1
            if n_chars >= budget["qa"]:
                break
        stats["qa"] = (n_chars,)
        print(f"qa: {n_chars / 1e6:.1f}M chars")

    total = os.path.getsize(out_path)
    print(f"\ntotal corpus: {total / 1e6:.1f} MB -> {out_path}")
    print("stats:", {k: v for k, v in stats.items()})


if __name__ == "__main__":
    main()
