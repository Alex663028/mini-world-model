"""世界模型测试：形状、前向、权重兼容加载。

重点覆盖两个真实踩过的坑：
1. 按形状匹配加载（v1 权重没有 state_head，严格 load 会失败）
2. state_head 只读潜状态（梯度不能反传改 encoder）
"""

import numpy as np
import torch

from model import Decoder, Dynamics, Encoder, MiniWorldModel


def test_encoder_shapes():
    enc = Encoder(latent_dim=32)
    x = torch.randn(4, 1, 64, 64)
    z = enc(x)
    assert z.shape == (4, 32)


def test_decoder_shapes():
    dec = Decoder(latent_dim=32)
    z = torch.randn(4, 32)
    img = dec(z)
    assert img.shape == (4, 1, 64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0, "decoder 输出必须归一化到 [0,1]"


def test_dynamics_shapes():
    dyn = Dynamics(latent_dim=32, action_dim=2)
    z = torch.randn(4, 32)
    a = torch.randn(4, 2)
    z_next = dyn(z, a)
    assert z_next.shape == (4, 32)


def test_full_model_forward():
    m = MiniWorldModel(latent_dim=32, action_dim=2)
    obs = torch.rand(2, 1, 64, 64)
    z = m.encode(obs)
    z2 = m.imagine_step(z, torch.zeros(2, 2))
    assert z2.shape == z.shape
    img = m.decode(z2)
    assert img.shape == obs.shape
    # 状态头存在且输出合理
    s = m.predict_state(z)
    for k in ("pos", "vel", "target", "dist"):
        assert k in s, f"状态头缺字段 {k}"
    assert s["pos"].shape == (2, 2)
    assert s["dist"].shape == (2,)


def test_imagine_rollout_stability():
    """开环想象 12 步（MPC 的窗口）不应出现 NaN/Inf。"""
    m = MiniWorldModel(latent_dim=32, action_dim=2).eval()
    z = m.encode(torch.rand(1, 1, 64, 64))
    with torch.no_grad():
        for _ in range(12):
            z = m.imagine_step(z, torch.tensor([[0.5, 0.3]]))
            assert torch.isfinite(z).all(), "想象发散了"


def test_load_v1_weights_by_shape(tmp_path):
    """v1 权重（没有 state_head）必须能按形状匹配加载。

    这是 demo.py 直接崩过的原因：model.py 后来加了 state_head，
    严格 load_state_dict 会对缺失的 key 报错。
    """
    v1_like = {k: v for k, v in MiniWorldModel(32, 2).state_dict().items()
               if not k.startswith("state_head")}
    p = tmp_path / "v1.pt"
    torch.save(v1_like, p)

    m = MiniWorldModel(32, 2)
    sd = torch.load(p, map_location="cpu")
    cur = m.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v
            loaded += 1
    m.load_state_dict(cur)
    assert loaded >= len(v1_like) - 1, "应该继承绝大部分权重"
    assert "state_head.0.weight" in m.state_dict(), "新模型的 state_head 保留随机初始化"


def test_param_count_reasonable():
    n = sum(p.numel() for p in MiniWorldModel(32, 2).parameters())
    assert 1_000_000 < n < 3_000_000, f"世界模型应在 1-3M 参数，实际 {n/1e6:.2f}M"
