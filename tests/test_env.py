"""环境测试：物理公式、边界反弹、到达判定、撞墙上报。

这些断言直接对应 agent.py 的事件记忆——物理公式或撞墙标志错了，
"你撞墙了吗"就会撒谎（实测踩过：36 步误报 24 次撞墙；
又反向踩过：反推公式的阈值永远不触发，撞墙事件一条都记不到）。
"""

import numpy as np

from env import ACTION_GAIN, DAMPING, REACH_DIST, PointMassEnv, random_policy


def test_reset_shapes():
    env = PointMassEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (64, 64), "观测必须是 64x64 灰度图"
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() and obs.max() <= 1.0, "像素值必须在 [0,1]"
    assert env.steps == 0
    assert np.all(np.abs(env.pos) <= 1.0)
    assert np.all(np.abs(env.target) <= 1.0)


def test_physics_formula():
    """核心断言：v' = 0.8v + 0.12a，pos' = pos + v' * 0.10。

    agent.py 的撞墙检测就依赖这个公式。写成 pos + 0.8v + 0.12a
    （漏掉 0.10 的位移系数）会导致每步都判撞墙。
    """
    env = PointMassEnv(seed=123)
    env.reset()
    # 把 agent 放到中间，远离边界，确保不会撞墙干扰
    env.pos = np.zeros(2, dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)
    a = np.array([0.5, -0.25], dtype=np.float32)

    v_expect = DAMPING * np.zeros(2) + ACTION_GAIN * a
    pos_expect = np.zeros(2) + v_expect * 0.10

    env.step(a)
    assert np.allclose(env.vel, v_expect, atol=1e-6), f"速度公式不符：{env.vel} vs {v_expect}"
    assert np.allclose(env.pos, pos_expect, atol=1e-6), f"位置公式不符：{env.pos} vs {pos_expect}"


def test_boundary_bounce():
    """撞墙后位置不能越界（世界是 ±1 的方框）。"""
    env = PointMassEnv(seed=7)
    env.reset()
    env.pos = np.array([0.98, 0.0], dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)
    for _ in range(20):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
        assert np.all(np.abs(env.pos) <= 1.02), f"跑出边界：{env.pos}"


def test_reach_and_timeout():
    """目标就在脚下 → 立刻 reached；一直不动 → 到 max_steps 终止。"""
    env = PointMassEnv(seed=1, max_steps=5)
    env.reset()
    env.target = env.pos.copy()
    _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
    assert done and info["reached"], "目标重合时应判定到达"
    assert not info["hit_wall"], "中间位置不应报撞墙"

    env2 = PointMassEnv(seed=2, max_steps=3)
    env2.reset()
    done = False
    for _ in range(3):
        _, _, done, info = env2.step(np.zeros(2, dtype=np.float32))
    assert done and not info["reached"], "超时应终止但不算到达"


def test_hit_wall_flag_reported_by_env():
    """撞墙必须由环境上报（agent 的事件记忆只读这个标志）。

    回归：agent 曾用"复制物理公式反推 + 偏差阈值 0.25"来判撞墙，
    而正常动作增益下偏差上限只有 ~0.08，导致撞墙事件一条都记不到。
    """
    env = PointMassEnv(seed=7)
    env.reset()
    env.pos = np.array([0.99, 0.0], dtype=np.float32)
    env.vel = np.zeros(2, dtype=np.float32)
    hit = False
    for _ in range(10):
        _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        hit = hit or info["hit_wall"]
    assert hit, "持续朝墙推必须上报 hit_wall"

    env2 = PointMassEnv(seed=8)
    env2.reset()
    env2.pos = np.zeros(2, dtype=np.float32)
    env2.vel = np.zeros(2, dtype=np.float32)
    _, _, _, info2 = env2.step(np.zeros(2, dtype=np.float32))
    assert info2["hit_wall"] is False, "场地中央不能误报撞墙"
    assert REACH_DIST == 0.06, "到达阈值改动需同步 README 数值"


def test_random_policy_range():
    """random_policy 是生成器（带惯性的平滑策略，动作幅度 ≤ 1）。"""
    rng = np.random.default_rng(0)
    gen = random_policy(rng)
    for _ in range(50):
        a = next(gen)
        assert a.shape == (2,)
        assert np.all(np.abs(a) <= 1.0)


def test_render_marks_agent_and_target():
    """渲染出的画面里应该有亮点（agent）和环（target）。"""
    env = PointMassEnv(seed=5)
    obs = env.reset()
    assert obs.max() > 0.5, "画面里必须有可见亮点"
    # agent 与目标重叠时，画面依然能渲染
    env.target = env.pos.copy()
    obs2 = env.render()
    assert obs2.shape == (64, 64)
