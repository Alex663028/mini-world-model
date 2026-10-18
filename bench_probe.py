"""性能探针：区分"AMD GPU 慢"还是"训练规模太大"。

2026-10-18 实测结论（RX 6800 + torch 2.13+rocm10.1）：

1. GPU 本身完全健康——真实配置（block 256 / seq 255 / fp16 autocast）下
   123.4M 模型 batch 48 只要 629 ms/step，和中午 104M pretrain 的
   1.28 s/step（batch 48/block 256）同量级。

2. **慢的真相是大 batch 下 ROCm 显存管理劣化**（不是算力）：
   batch  48:   629 ms/step  健康
   batch  96:  5549 ms/step  劣化 9 倍
   batch 128: 13642 ms/step  劣化 22 倍
   且 ms/样本随 batch 上升（57.8 / 106.6 vs 13.1）——rocBLAS/allocator
   在显存压力下走慢路径，而非算力瓶颈。

3. fp32 是陷阱：RDNA2 的 fp32 在 ROCm 上不走矩阵加速，
   2048³ 矩阵乘实测 0.5 TFLOPS（理论 18）。必须 fp16 autocast
   （train_bpe/train_lm_state/train_sft 都已用 GradScaler）。

4. 探针第一版的错误：BLOCK 写成 512，而 train_bpe.py 的真实 BLOCK 是
   256——seq 511 vs 255 差 4 倍 attention 计算量，导致"GPU 慢 20 倍"
  的误判。**探针必须先对齐真实配置再下结论**。

用法：python bench_probe.py
"""

import sys
import time

import torch

sys.path.insert(0, ".")
from tinygpt import TinyGPT, count_params

BLOCK = 256  # 与 train_bpe.py 对齐（BPE token 上下文）
BATCH = 48   # 探针验证的健康 batch


def bench_step_fp16(model, batch, block, repeat=5):
    """一次完整 train step（fp16 autocast + GradScaler，与真实脚本一致）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randint(0, 6401, (batch, block - 1), device="cuda")
    y = torch.randint(0, 6401, (batch, block - 1), device="cuda")
    for _ in range(2):  # warmup
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return (time.time() - t0) / repeat


def main():
    print("=" * 60)
    print("性能探针（RX 6800 / ROCm）——配置已对齐 train_bpe.py")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return

    print(f"\n[1] 123M 模型（16层/768）batch scaling（block {BLOCK}, fp16）")
    m = TinyGPT(vocab_size=6401, d_model=768, n_heads=12, n_layers=16,
                block_size=BLOCK - 1, state_dim=0).cuda()
    for bs in (48, 64, 96, 128):
        try:
            t = bench_step_fp16(m, bs, BLOCK)
            print(f"  batch {bs:4d}: {t * 1000:7.1f} ms/step ({t / bs * 1000:6.1f} ms/样本)")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:4d}: OOM")
            torch.cuda.empty_cache()
    del m
    torch.cuda.empty_cache()

    print(f"\n[2] 模型规模 scaling（batch {BATCH}, fp16）")
    for name, d, layers, heads in [
        ("26M(8层/512)", 512, 8, 8),
        ("75M(14层/640)", 640, 14, 10),
        ("123M(16层/768)", 768, 16, 12),
    ]:
        m = TinyGPT(vocab_size=6401, d_model=d, n_heads=heads, n_layers=layers,
                    block_size=BLOCK - 1, state_dim=0).cuda()
        t = bench_step_fp16(m, BATCH, BLOCK)
        print(f"  {name:18s} {count_params(m) / 1e6:5.1f}M  {t * 1000:7.1f} ms/step")
        del m
        torch.cuda.empty_cache()

    print("\n判读：")
    print("  - batch 48 健康（ms/样本 ~13）→ 用 48，有效 batch 靠 grad accum 补")
    print("  - ms/样本随 batch 上升 = ROCm allocator 劣化，不要靠加 batch 提速")
    print("  - 任何'GPU 慢'的结论先核对 BLOCK/autocast 是否对齐真实训练脚本")


if __name__ == "__main__":
    main()
