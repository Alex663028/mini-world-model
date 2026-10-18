"""lm_state_bpe 验证：BPE 104M + 潜状态注入的感知能力。

用法：python verify_state_bpe.py [--ckpt lm_state_bpe.pt]

对比三个模型的感知问答：
  A. char 版 lm_state_motion（75M 生产模型）
  B. BPE 版 lm_state_bpe（123M + 潜状态注入）
同一批 env 帧、同一批问题，看谁答得准。
"""

import argparse
import os
import sys

import numpy as np
import torch

# 代码从脚本所在目录 import；权重按 CWD 找（norm 是权重和脚本同目录跑）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import PointMassEnv
from model import MiniWorldModel
from tinygpt import TinyGPT
from tokenizer import load_tokenizer

QUESTIONS = [
    "你看到了什么？",
    "目标离你有多远？",
    "你现在在往哪边走？",
    "你和目标的方位关系是什么？",
]


def load_lm(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=True)
    c = ck["config"]
    tok_name = c.get("tokenizer", "char")
    if tok_name.endswith(".json"):
        tok = load_tokenizer(tok_name)
        vocab_size = len(ck["stoi"])   # stoi 已含 [STATE]（train_lm_state 追加过）
        enc, dec = lambda t: tok.encode(t), lambda ids: tok.decode(ids)
    else:
        tok = None
        vocab_size = len(ck["stoi"])
        enc = lambda t: [ck["stoi"][ch] for ch in t if ch in ck["stoi"]]
        dec = lambda ids: "".join(ck["itos"][i] for i in ids if i < len(ck["itos"]))
    m = TinyGPT(vocab_size=vocab_size, d_model=c["d_model"], n_heads=c["n_heads"],
                n_layers=c["n_layers"], block_size=c["block_size"],
                state_dim=c.get("state_dim", 0), state_token_id=c.get("state_token_id"))
    sd = ck["model"]
    cur = m.state_dict()
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    m.load_state_dict(cur)
    m.to(device).eval()
    return m, c, enc, dec


def ask(model, cfg, enc, dec, question, z, state_id, device="cuda", max_new=48):
    prompt = f"问：{question}\n{chr(0xE000)}\n答："
    ids = enc(prompt)
    if state_id not in ids:  # BPE 可能把 STATE 编进长 token，强制插入
        ids = enc(f"问：{question}\n") + [state_id] + enc("\n答：")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    st = None
    if getattr(model, "state_dim", 0) > 0:
        # 按模型 state_dim 对齐：32 = 纯 z；64 = z + Δz（单帧无 Δz，补零）
        want = model.state_dim
        zz = np.asarray(z, dtype=np.float32).reshape(-1)
        if zz.size < want:
            zz = np.concatenate([zz, np.zeros(want - zz.size, dtype=np.float32)])
        st = torch.tensor([zz[:want]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, temperature=0.3,
                             top_k=10, state=st)
    full = dec(out[0].tolist())
    gen = full[len(dec(ids)):]          # 只保留新生成部分（去掉 prompt）
    for sep in ["\n", "问：", "系统："]:
        if sep in gen:
            gen = gen.split(sep)[0]
    return gen.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lm_state_bpe.pt")
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wm = MiniWorldModel(latent_dim=32, action_dim=2)
    if not os.path.exists("checkpoint.pt"):
        raise SystemExit(
            "[verify] 找不到世界模型权重 checkpoint.pt（按当前目录找）。\n"
            "         先训练：python train.py --episodes 400 --epochs 20\n"
            "         或从 Release 下载 checkpoint.pt 放到当前目录。"
        )
    sd = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
    cur = wm.state_dict()   # v1 权重没有 state_head（v3 才加），按形状匹配加载
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
    wm.load_state_dict(cur)
    wm.to(device).eval()

    results, hit_counts = {}, {}
    for name, path in [("B-BPE104M", args.ckpt),
                       ("A-char75M", "lm_state_motion.pt")]:
        if not os.path.exists(path):
            print(f"{name}: {path} 不存在，跳过")
            continue
        m, c, enc, dec = load_lm(path, device)
        state_id = c.get("state_token_id", 0)
        print(f"\n{'=' * 56}\n{name}（{path}, {c['d_model']}维/{c['n_layers']}层, "
              f"state_dim={c.get('state_dim', 0)}）\n{'=' * 56}")
        ok, used = 0, 0
        for s in range(args.seeds):
            env = PointMassEnv(seed=2026 + s * 977)
            env.reset()
            obs = env.render()
            with torch.no_grad():
                z = wm.encode(torch.from_numpy(
                    obs[None, None].astype(np.float32)).to(device)).reshape(-1)[:32].cpu().numpy()
            true_d = float(np.linalg.norm(env.pos - env.target))
            print(f"\n-- seed {2026 + s * 977}: 真实距离 {true_d:.2f} --")
            for q in QUESTIONS:
                a = ask(m, c, enc, dec, q, z, state_id, device)
                print(f"  {q[:14]:16s} -> {a[:52]}")
                if "远" in q and any(f"{true_d:.1f}" in a for _ in [0]):
                    ok += 1
            used += 1
        results[name] = ok
        hit_counts[name] = used
        # 分母用实际跑过的 seed 数：原来写死 /4，--seeds 16 时输出也是 x/4
        print(f"\n{name} 距离问答命中: {ok}/{used}")

    if len(results) == 2:
        print("\n" + "=" * 56)
        a, b = "A-char75M", "B-BPE104M"
        print(f"对比: {a} {results[a]}/{hit_counts[a]}  vs  "
              f"{b} {results[b]}/{hit_counts[b]}")


if __name__ == "__main__":
    main()
