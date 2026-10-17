"""MiniWorldAgent：所有模块融合的统一入口。

这是整个系统的总装——一个类收拢全部能力：

    感知层   世界模型（编码器 / 动力学 / 解码器）+ 程序化环境
    语言层   TinyGPT（L2 潜状态注入生成）+ 常识库检索（兜底）
    行动层   MPC 想象规划 + PD 逆动力学精密进近
    符号层   四则运算计算器（语言模型不碰算术）
    记忆层   对话历史 + 环境状态

统一路由（reply 单函数）：
    1. 算术            → 符号计算器
    2. 任务 / 环境操控 / 边界 → 世界模型行动层
    3. 感知类          → 64 维潜状态注入生成（当前 z + 一步预见）
    4. 常识类          → 语言模型生成 + 检索兜底（高置信命中时补充权威答案）
    5. 闲聊 / 其他     → 语言模型生成

用法：
    from agent import MiniWorldAgent
    bot = MiniWorldAgent()
    bot.reply("去目标那儿")
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat import bigrams, extract_agent_pos
from corpus_dialog import (BOUND_Q, BYE, BYE_Q, CAN, CAN_Q, FAIL_Q, GREET, GREET_Q,
                           NO, NO_Q, TASK_Q, TASK_READY, THANKS, THANKS_Q,
                           WHO, WHO_Q, YES, YES_Q)
from env import PointMassEnv
from knowledge import ALL_KNOWLEDGE
from model import MiniWorldModel
from tinygpt import TinyGPT

# 常识索引打通大库（580 条）+ 小库（40 条）——
# 曾经索引只建了小库，大库写了却没用上（审核发现的数据断点）
from science_big import SCIENCE_BIG
from history_geo_big import HISTORY_BIG
from tech_big import TECH_BIG
from life_big import LIFE_BIG


def _iter_kb_entries():
    """统一遍历常识库：小库是 dict，大库是 (key, fact) 元组。"""
    for entry in ALL_KNOWLEDGE:
        yield entry["keys"], entry["answer"]
    for group in (SCIENCE_BIG, HISTORY_BIG, TECH_BIG, LIFE_BIG):
        for key, fact in group:
            yield [key], fact

STATE_TOKEN = "\uE000"
PROMPT_END = "\n答："

# ---- 可配置参数（原魔法数字集中管理）----
CONFIG = {
    "seed": 2026,
    "n_candidates": 8,        # MPC 候选方向数
    "imagine_horizon": 12,    # 世界模型想象步数
    "task_budget": 60,        # 导航任务步数预算
    "task_timeout": 90,       # 导航任务超时（秒）
    "approach_dist": 0.4,     # 切换控制层的距离阈值
    "stuck_limit": 3,         # MPC 无进展让步步数
    "report_every": 15,       # 进度汇报间隔（步）
    "max_tokens": 80,         # 生成最大 token
    "temperature": 0.5,
    "top_k": 20,
    "history_window": 6,      # 多轮上下文窗口（轮）
    "kb_threshold": 0.45,     # 检索命中阈值
}

# 各类意图的关键词路由（按优先级排列，具体词优先于宽泛词）
TASK_KEYS = ["去目标", "走到目标", "导航", "前往", "到达目标", "去那边", "找目标", "追踪", "走过去"]
RESET_KEYS = ["重置", "换一个目标", "换目标", "新目标", "重新开始", "新回合", "再来"]
BOUNDARY_KEYS = ["墙在哪", "边界", "会撞墙", "四周", "边上有"]
PERCEPTION_KEYS = ["看到", "眼前", "画面", "观测", "在哪", "哪个方向", "什么方向",
                   "多远", "方位", "位置", "周围", "环境", "多快", "速度"]
IMAGINE_KEYS = ["想象一下", "想象", "梦境", "推演", "rollout"]
ACT_KEYS = ["往前走", "向前走", "往左", "向左", "往右", "向右", "往后退", "推一下", "走一步"]
LATENT_KEYS = ["潜状态", "latent", "编码", "表征"]

DIR8 = ["正东", "东北", "正北", "西北", "正西", "西南", "正南", "东南"]


class MiniWorldAgent:
    def __init__(self, world_ckpt="checkpoint.pt", lm_ckpt="lm_state.pt",
                 dialog_ckpt="lm_unified.pt", seed=None, config=None):
        cfg = dict(CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg
        seed = seed if seed is not None else cfg["seed"]

        # ---- 设备统一：世界模型与语言模型同设备（消除 CPU-GPU 跨设备拷贝）----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dz_scale = 1.0  # Δz 缩放（LM config 会覆盖为训练时的值）

        # ---- 感知层：世界模型 ----
        self.model = MiniWorldModel(latent_dim=32, action_dim=2)
        self.model.load_state_dict(torch.load(world_ckpt, map_location=self.device))
        self.model.to(self.device)  # 权重映射不移动模型本体，必须显式 .to()
        self.model.eval()
        self.env = PointMassEnv(seed=seed)
        self.env.reset()

        # ---- 语言层：双模型分工 ----
        # lm = 具身感知模型（潜状态注入，回答"你在哪/看到什么"）——世界模型的"眼睛"
        # lm_dialog = 三域文本模型（对话 + 知识 + 世界叙述，混合语料专精）
        # 分工的依据：一个模型干两件事两头都不灵——感知靠潜状态注入，
        # 纯文本域（闲聊/知识/叙述）靠混合语料，两者互补而非替代
        loaded = self._load_lm(lm_ckpt)
        if loaded:
            self.lm, self.stoi, self.itos, lm_cfg = loaded
            self.state_dim = lm_cfg.get("state_dim", 0)
            self.state_token_id = lm_cfg.get("state_token_id")
            self.dz_scale = lm_cfg.get("dz_scale", 1.0)
        else:
            self.lm, self.state_dim, self.state_token_id = None, 0, None
        dlg = self._load_lm(dialog_ckpt)
        self.lm_dialog = dlg[0] if dlg else None
        self.lm_dialog_stoi = dlg[1] if dlg else None
        self.lm_dialog_itos = dlg[2] if dlg else None

        # ---- 常识检索索引（大库 + 小库全量打通）----
        self.kb = []
        for keys, answer in _iter_kb_entries():
            bg = set()
            for key in keys:
                bg |= bigrams(key)
            if bg:
                self.kb.append((bg, {"keys": keys, "answer": answer}))

        # ---- 记忆层 + 并发锁（HTTP 多客户端共享实例时保护 env 状态）----
        self.history = []
        self.last_z = None  # 上一帧潜状态（运动感知 Δz 用）
        self._lock = threading.Lock()

    def _load_lm(self, path):
        """加载一个 TinyGPT checkpoint，返回 (model, stoi, itos, config) 或 None。"""
        if not path or not os.path.exists(path):
            return None
        try:
            ck = torch.load(path, map_location=self.device)
            c = ck["config"]
            model = TinyGPT(
                vocab_size=len(ck["stoi"]),
                d_model=c["d_model"], n_heads=c["n_heads"],
                n_layers=c["n_layers"], block_size=c["block_size"],
                state_dim=c.get("state_dim", 0), state_token_id=c.get("state_token_id"),
            )
            model.load_state_dict(ck["model"])
            model.to(self.device)  # 同上：显式移动模型本体
            model.eval()
            return model, ck["stoi"], ck["itos"], c
        except Exception as e:
            sys.stderr.write(f"[agent] LM load failed ({path}: {e})\n")
            return None

    # ==================== 主入口：统一路由 ====================

    # ==================== Agentic 模式：模型自主决定调工具 ====================

    SFT_SYSTEM = (
        "你是小WM，一个由世界模型和语言模型组成的迷你 agent。"
        "你能看见 64×64 灰度画面里的自己和目标环、能规划路径走过去、"
        "能查询常识、能做四则运算。需要这些能力时，你会调用对应工具。"
    )

    def _sft_generate(self, prompt, max_tokens=90, temperature=0.3, top_k=10):
        """用 SFT 模型生成（conversations 格式：系统/用户/助手）。"""
        if self.lm_dialog is None:
            return ""
        idx = torch.tensor(
            [[self.lm_dialog_stoi[c] for c in prompt if c in self.lm_dialog_stoi]],
            dtype=torch.long, device=self.device,
        )
        nl = {self.lm_dialog_stoi["\n"]} if "\n" in self.lm_dialog_stoi else None
        with torch.no_grad():
            out = self.lm_dialog.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature,
                top_k=top_k, stop_ids=nl,
            )
        return "".join(self.lm_dialog_itos[i] for i in out[0].tolist())[len(prompt):].strip()

    def execute_tool(self, name, question=""):
        """执行工具调用——agentic 模式与路由模式共用同一套工具实现。"""
        if name == "world_perceive":
            return self.reply("你看到了什么？")
        if name == "world_navigate":
            return self._run_navigation()
        if name == "world_imagine":
            return self._imagine("想象一下往左走")
        if name == "world_act":
            return self._act("往前走")
        if name == "kb_query":
            hit = self._retrieve(question)
            return hit[0] if hit else "常识库没有这条"
        if name == "calc":
            return self._try_calc(question) or "算式没解析出来"
        return f"未知工具 {name}"

    @staticmethod
    def _parse_tool(text):
        """从生成文本里解析 [调用工具 name]。"""
        if "[调用工具" not in text:
            return None, ""
        body = text.split("[调用工具", 1)[1]
        name = body.split("]")[0].strip()
        return name, text.split("]", 1)[1] if "]" in text else ""

    def reply_agentic(self, text, max_rounds=3):
        """模型驱动的工具调用循环（MiniMind Agentic 模式）。

        与路由模式的区别：不是关键词规则决定调哪个工具，而是 SFT 模型
        自己输出工具调用，agent 执行后把结果回灌，让模型总结成自然语言。

        注意：路由模式在感知精确度上仍更稳（潜状态直读），
        agentic 模式展示的是"模型自主决策工具"的能力。
        """
        prompt = f"系统：{self.SFT_SYSTEM}\n用户：{text}\n助手："
        for _ in range(max_rounds):
            gen = self._sft_generate(prompt)
            if not gen:
                break
            tool, _ = self._parse_tool(gen)
            if not tool:
                return gen
            result = self.execute_tool(tool, text)
            prompt += f"{gen}\n结果：{result}\n助手："
        return self._sft_generate(prompt)

    def reply(self, text):
        """统一入口。带错误降级：任何子能力失败不影响其他能力。"""
        text = text.strip()
        if not text:
            return "说点什么吧。"
        try:
            answer = self._route(text)
        except Exception as e:
            sys.stderr.write(f"[agent] reply failed on '{text[:30]}': {e}\n")
            answer = f"内部出了点问题（{type(e).__name__}），换个问法或说'重置'试试。"
        self._finish(text, answer)
        return answer

    def _route(self, text):
        self.history.append(("user", text))

        # 1. 算术 → 符号计算器
        calc = self._try_calc(text)
        if calc is not None:
            return calc

        # 2. 任务 / 环境操控 / 边界 → 世界模型行动层
        if any(k in text for k in RESET_KEYS):
            return self._do_reset()
        if any(k in text for k in TASK_KEYS):
            return self._run_navigation()
        if any(k in text for k in BOUNDARY_KEYS):
            return self._describe_boundary()

        # 3. 感知类 → 潜状态注入生成
        if self.lm is not None and self.state_dim > 0 and any(k in text for k in PERCEPTION_KEYS):
            return self._perception_answer(text)

        # 4. 想象 / 单步动作 / 潜状态统计 → 世界模型工具查询
        if any(k in text for k in IMAGINE_KEYS):
            return self._imagine(text)
        if any(k in text for k in ACT_KEYS):
            return self._act(text)
        if any(k in text for k in LATENT_KEYS):
            return self._latent_report()

        # 5. 常识 / 闲聊 → 检索优先，生成兜底
        return self._chat_answer(text)

    def _finish(self, question, answer):
        self.history.append(("bot", answer))
        if len(self.history) > 20:
            del self.history[:2]
        return answer

    # ==================== 感知层：64 维潜状态 ====================

    def perceive(self):
        """编码当前观测，产出注入语言模型的状态（GPU 同设备）。

        融合架构 v2（运动感知）：
        - state_dim = 32：只给当前帧潜状态
        - state_dim = 64：当前 z ⊕ Δz（与上一帧潜状态之差 = 运动特征，
          语言模型因此能感知"我在怎么动"，而不只是静态快照）
        """
        obs = self.env.render()
        obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            z = self.model.encode(obs_t)
            if self.state_dim >= 64:
                dz = torch.zeros_like(z) if self.last_z is None else z - self.last_z
                self.last_z = z.detach()
                return torch.cat([z, dz * self.dz_scale], dim=1)
        return z

    def _perception_answer(self, question):
        if self.lm is None:
            return self._fallback_perception()
        state = self.perceive()
        return self._generate(question, state=state, use_ctx=False)  # 感知通道不带历史

    def _fallback_perception(self):
        pos, target = self.env.pos, self.env.target
        return (
            f"我在 ({pos[0]:.2f}, {pos[1]:.2f})，目标在 ({target[0]:.2f}, {target[1]:.2f})，"
            f"直线距离 {float(np.linalg.norm(pos - target)):.1f}。"
        )

    # ==================== 语言层 ====================

    def _encode(self, s):
        return torch.tensor([[self.stoi[c] for c in s if c in self.stoi]], dtype=torch.long,
                            device=self.device)

    def _generate_dialog(self, question, max_tokens=80, temperature=None, top_k=None):
        """对话模型生成（闲聊兜底）。专用对话模型：见过 385 条高质量对话语料，
        输出是"对话模式"而非"百科模式"。"""
        if self.lm_dialog is None:
            return ""
        cfg = self.cfg
        temperature = cfg["temperature"] if temperature is None else temperature
        top_k = top_k or cfg["top_k"]
        prompt = f"问：{question}{PROMPT_END}"
        idx = torch.tensor(
            [[self.lm_dialog_stoi[c] for c in prompt if c in self.lm_dialog_stoi]],
            dtype=torch.long, device=self.device,
        )
        stop_ids = {self.lm_dialog_stoi[c] for c in "\n" if c in self.lm_dialog_stoi}
        with torch.no_grad():
            out = self.lm_dialog.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature,
                top_k=top_k, stop_ids=stop_ids,
            )
        gen = "".join(self.lm_dialog_itos[i] for i in out[0].tolist())[len(prompt):]
        return gen.strip().rstrip("问：答").strip("： ")

    def _generate(self, question, state=None, max_tokens=None, temperature=None, top_k=None,
                  use_ctx=True):
        """语言模型生成；state 非空时插入 [STATE] token 并注入潜状态。

        use_ctx=False 时不带多轮上下文——感知通道必须关掉它：
        历史问答（含知识域的长答案）会挤掉 [STATE] 的位置，
        实测导致 state 注入失效、回答退化成百科幻觉。
        """
        cfg = self.cfg
        max_tokens = max_tokens or cfg["max_tokens"]
        temperature = cfg["temperature"] if temperature is None else temperature
        top_k = top_k or cfg["top_k"]

        # 多轮上下文（最近 N 轮，含本轮问题）
        ctx = ""
        if use_ctx:
            recent = [h for h in self.history[-(cfg["history_window"] * 2):]]
            for role, msg in recent:
                if role == "user":
                    ctx += f"问：{msg}\n"
                else:
                    ctx += f"答：{msg[:40]}\n"
        if state is not None and self.state_token_id is not None:
            prompt = f"{ctx}问：{question}\n{STATE_TOKEN}{PROMPT_END}"
        else:
            prompt = f"{ctx}问：{question}{PROMPT_END}"
        idx = self._encode(prompt)
        stop_ids = {self.stoi[c] for c in "\n" if c in self.stoi}
        with torch.no_grad():
            out = self.lm.generate(
                idx, max_new_tokens=max_tokens, temperature=temperature,
                top_k=top_k, stop_ids=stop_ids, state=state,
            )
        gen = "".join(self.itos[i] for i in out[0].tolist())[len(prompt):]
        return gen.strip().rstrip("问：答").strip("： ")

    def _retrieve(self, question, threshold=None):
        """常识库检索（查询覆盖率：查询的 bigram 被条目覆盖的比例）。

        用查询侧归一（|交集|/|查询|）而不是 min(|查询|,|条目|)——
        后者会让短查询吃亏（"长城"对"万里长城"覆盖率 100% 却只得 0.33）。
        """
        if threshold is None:
            threshold = self.cfg["kb_threshold"]
        q_bg = bigrams(question)
        if not q_bg:
            return None
        best, best_score = None, 0.0
        for entry_bg, entry in self.kb:
            if not entry_bg:
                continue
            score = len(q_bg & entry_bg) / len(q_bg)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= threshold:
            return best["answer"], best_score
        return None

    def _system_answer(self, text):
        """系统自述精确回答：从对话语料结构化数据查表，零幻觉、可控。

        为什么要这层：对话模型专精"问什么答什么"后，对身份/能力类问题
        会稳定地自报家门（语料里这类占 40%），但对"谢谢你"也报家门就串了。
        这类问题本质是固定知识，直接查表比生成可靠。

        匹配策略：所有问法合并成一张表、按长度降序——
        避免短问法劫持长问法（"你会什么" 抢走 "你会什么语言"）。
        """
        t = text.rstrip("？?。.!！")
        table = []
        for q_list, a_list in [
            (WHO_Q, WHO), (CAN_Q, CAN), (GREET_Q, GREET),
            (THANKS_Q, THANKS), (BYE_Q, BYE), (YES_Q, YES), (NO_Q, NO),
            (TASK_Q, TASK_READY),
        ]:
            table.extend((q, a_list) for q in q_list)
        table.extend((q, [a]) for q, a in BOUND_Q + FAIL_Q)
        table.sort(key=lambda x: -len(x[0]))
        for q, a_list in table:
            if t == q or (len(q) >= 3 and q in text):
                return a_list[len(text) % len(a_list)]
        return None

    def _chat_answer(self, question):
        """常识/闲聊：四层兜底——系统查表 → 知识检索 → 对话生成 → 诚实说不知道。"""
        # 1. 系统自述/能力/礼貌（精确，零幻觉）
        sys_ans = self._system_answer(question)
        if sys_ans:
            return sys_ans
        # 2. 常识库检索
        hit = self._retrieve(question, threshold=self.cfg["kb_threshold"])
        if hit:
            return hit[0]
        # 3. 对话模型生成（闲聊兜底）
        if self.lm_dialog is not None:
            gen = self._generate_dialog(question)
            if gen:
                return gen
        if self.lm is not None:
            gen = self._generate(question)
            if gen:
                return gen
        return "这个我答不上来。我的常识库有六百多条，但这条不在里面。"

    # ==================== 符号层 ====================

    def _try_calc(self, text):
        m = re.search(
            r"(-?\d+(?:\.\d+)?)\s*(乘以|除以|加上|减去|\+|\-|\*|x|X|×|÷)\s*(-?\d+(?:\.\d+)?)",
            text,
        )
        if not m:
            return None
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
        if op in ("+", "加上"):
            r = a + b
        elif op in ("-", "减去"):
            r = a - b
        elif op in ("*", "x", "X", "×", "乘以"):
            r = a * b
        elif op in ("/", "÷", "除以"):
            if b == 0:
                return "除数不能为 0，这个我没法算。"
            r = a / b
        else:
            return None
        if abs(r - round(r)) < 1e-9:
            r = round(r)
        return f"{a:g} {op} {b:g} = {r:g}。算术走符号计算器，不走语言模型。"

    # ==================== 行动层：MPC + PD ====================

    def _bearing(self, frm, to):
        dx, dy = to - frm
        ang = math.degrees(math.atan2(-dy, dx))
        return DIR8[int(((ang + 22.5) % 360) // 45)]

    def _run_navigation(self):
        """语言即任务：MPC 规划 + PD 控制分层，带进度汇报与超时保护。

        并发安全：整个导航持锁执行，避免 HTTP 多客户端争用 env 状态。
        """
        cfg = self.cfg
        with self._lock:
            return self._run_navigation_locked()

    def _run_navigation_locked(self):
        cfg = self.cfg
        reports = []
        steps_used, switches, phase = 0, 0, "mpc"
        prev_a = np.zeros(2, dtype=np.float32)
        stuck = 0
        t0 = time.time()
        info = {"reached": False, "dist": float(np.linalg.norm(self.env.pos - self.env.target))}
        with torch.no_grad():
            for step in range(cfg["task_budget"]):
                if time.time() - t0 > cfg["task_timeout"]:
                    reports.append("任务超时中断")
                    break
                d0 = float(np.linalg.norm(self.env.pos - self.env.target))
                obs = self.env.render()
                obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(self.device)
                z = self.model.encode(obs_t)

                if d0 > cfg["approach_dist"] and stuck < cfg["stuck_limit"]:
                    # 规划层：MPC 想象（世界模型的主场）
                    throttle = min(0.7, max(0.3, d0 * 1.2))
                    candidates = [
                        np.array([np.cos(a), np.sin(a)], dtype=np.float32) * throttle
                        for a in np.linspace(0, 2 * np.pi, cfg["n_candidates"], endpoint=False)
                    ]
                    candidates.append((prev_a * 0.8).astype(np.float32))
                    candidates.append(np.zeros(2, dtype=np.float32))
                    best_a, best_pd = None, np.inf
                    for a in candidates:
                        zc = z
                        min_d = np.inf
                        for _ in range(cfg["imagine_horizon"]):
                            zc = self.model.imagine_step(zc, torch.from_numpy(a).unsqueeze(0).to(self.device))
                            p = self._safe_pos(self.model.decode(zc).squeeze().cpu().numpy())
                            if p is not None:
                                min_d = min(min_d, float(np.linalg.norm(p - self.env.target)))
                        if min_d < best_pd:
                            best_a, best_pd = a, min_d
                else:
                    # 控制层：PD 逆动力学精密进近
                    vec = self.env.target - self.env.pos
                    n = np.linalg.norm(vec)
                    if n > 1e-6:
                        desired_v = vec / n * min(0.3, n * 1.2)
                        a_ctrl = (desired_v - 0.8 * self.env.vel) / 0.12
                        best_a = np.clip(a_ctrl, -0.9, 0.9).astype(np.float32)
                    else:
                        best_a = np.zeros(2, dtype=np.float32)
                    if phase == "mpc":
                        phase = "approach"
                        switches += 1

                obs, _, done, info = self.env.step(best_a)
                prev_a = best_a
                steps_used = step + 1
                stuck = stuck + 1 if info["dist"] >= d0 - 0.01 else 0
                if steps_used % cfg["report_every"] == 0 and not done:
                    reports.append(
                        f"第{steps_used}步向{self._bearing(self.env.pos, self.env.target)}推进，还剩{info['dist']:.1f}"
                    )
                if done:
                    break
        if info.get("reached"):
            tail = f"过程：{'；'.join(reports)}。" if reports else ""
            note = f"中途切换控制层 {switches} 次。" if switches else ""
            return f"任务完成！用 {steps_used} 步到达目标。{note}{tail}"
        tail = f"过程：{'；'.join(reports)}。" if reports else ""
        return f"{steps_used} 步未能到达，最近时只剩 {info['dist']:.1f}。{tail}"

    @staticmethod
    def _safe_pos(img):
        """从想象帧提取 agent 位置，带 sanity check。

        argmax 提取在 agent 褪色（想象失真）时会静默抓到目标环/边界上。
        校验：提取位置必须在世界坐标合理范围内，否则返回 None（该候选作废）。
        """
        p = extract_agent_pos(img)
        if not np.all(np.isfinite(p)):
            return None
        if np.any(np.abs(p) > 1.05):  # 世界边界 ±1，留 5% 余量
            return None
        return p

    def _act(self, text):
        """单步动作执行。"""
        if "左" in text or "西" in text:
            a = np.array([-0.7, 0.0], dtype=np.float32)
        elif "右" in text or "东" in text:
            a = np.array([0.7, 0.0], dtype=np.float32)
        elif "上" in text or "前" in text or "北" in text:
            a = np.array([0.0, -0.7], dtype=np.float32)
        elif "下" in text or "后" in text or "南" in text:
            a = np.array([0.0, 0.7], dtype=np.float32)
        else:
            d = self.env.target - self.env.pos
            n = np.linalg.norm(d)
            a = (d / n * 0.7).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)
        obs, _, done, info = self.env.step(a)
        tail = " 我到达目标了！" if info["reached"] else ""
        return (
            f"执行了动作 ({a[0]:.1f}, {a[1]:.1f})，现在在 "
            f"({self.env.pos[0]:.2f}, {self.env.pos[1]:.2f})，离目标 {info['dist']:.2f}。{tail}"
        )

    def _imagine(self, text):
        """世界模型想象 rollout，报告梦里终点。"""
        action = self._parse_direction(text)
        obs = self.env.render()
        obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            z = self.model.encode(obs_t)
            zc = z
            for _ in range(self.cfg["imagine_horizon"]):
                zc = self.model.imagine_step(
                    zc, torch.from_numpy(action.astype(np.float32)).unsqueeze(0).to(self.device)
                )
            img = self.model.decode(zc).squeeze().cpu().numpy()
            p = self._safe_pos(img)
        if p is None:
            return "这次想象失真了（梦里的我跑出了世界边界），换个方向再试。"
        return (
            f"我想象了一下：朝那个方向推 {self.cfg['imagine_horizon']} 步，梦里的我会出现在 "
            f"({p[0]:.2f}, {p[1]:.2f})，离目标 {np.linalg.norm(p - self.env.target):.2f}。"
            "想象越远误差越大，12 步不算短了。"
        )

    def _parse_direction(self, text):
        if "左" in text or "西" in text:
            return np.array([-0.7, 0.0], dtype=np.float32)
        if "右" in text or "东" in text:
            return np.array([0.7, 0.0], dtype=np.float32)
        if "上" in text or "前" in text or "北" in text:
            return np.array([0.0, -0.7], dtype=np.float32)
        if "下" in text or "后" in text or "南" in text:
            return np.array([0.0, 0.7], dtype=np.float32)
        d = self.env.target - self.env.pos
        n = np.linalg.norm(d)
        return (d / n * 0.7).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)

    def _latent_report(self):
        obs = self.env.render()
        obs_t = torch.from_numpy(obs[None, None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            z = self.model.encode(obs_t)
        return (
            f"当前潜状态 32 维，均值 {z.mean().item():+.3f}，标准差 {z.std().item():.3f}。"
            "注入语言模型的是 32 维当前感知（64 维预见版实验后回退，存档 lm_state64.pt）。"
        )

    def _do_reset(self):
        self.env.reset()
        self.last_z = None  # 清空运动记忆（新回合）
        return (
            f"新回合开始。我在 ({self.env.pos[0]:.2f}, {self.env.pos[1]:.2f})，"
            f"目标在 ({self.env.target[0]:.2f}, {self.env.target[1]:.2f})，"
            f"直线距离 {float(np.linalg.norm(self.env.pos - self.env.target)):.2f}。"
        )

    def _describe_boundary(self):
        return (
            "我的世界四面都有墙：上下左右各一道两像素宽的暗灰边界，图上看得见。"
            "撞墙会反弹且速度减半，所以贴墙时我的想象力会下降——MPC 只信十二步以内的梦。"
        )


def main():
    """CLI 入口：单次调用 + JSON 输出。

    用法：
        python agent.py "去目标那儿"
        python agent.py "你看到了什么？" --json
        python agent.py --state          # 只看环境状态
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="MiniWorldAgent CLI")
    parser.add_argument("text", nargs="*", help="要说的内容（空则进入交互）")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--state", action="store_true", help="输出环境状态后退出")
    parser.add_argument("--lm", default="lm_state.pt", help="感知语言模型 checkpoint")
    parser.add_argument("--dialog", default="lm_unified.pt", help="对话语言模型 checkpoint")
    parser.add_argument("--agentic", action="store_true",
                        help="agentic 模式：由 SFT 模型自主决定调用哪个工具（需配 lm_sft2.pt）")
    args = parser.parse_args()

    bot = MiniWorldAgent(lm_ckpt=args.lm, dialog_ckpt=args.dialog)
    text = " ".join(args.text).strip()

    if args.state:
        state = {
            "agent_pos": bot.env.pos.tolist(),
            "target_pos": bot.env.target.tolist(),
            "distance": round(float(np.linalg.norm(bot.env.pos - bot.env.target)), 3),
            "steps": bot.env.steps,
            "lm_loaded": bot.lm is not None,
            "state_dim": bot.state_dim,
        }
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if text:
        reply = bot.reply_agentic(text) if args.agentic else bot.reply(text)
        if args.json:
            print(json.dumps({
                "input": text,
                "reply": reply,
                "agent_pos": bot.env.pos.tolist(),
                "target_pos": bot.env.target.tolist(),
                "distance": round(float(np.linalg.norm(bot.env.pos - bot.env.target)), 3),
            }, ensure_ascii=False, indent=2))
        else:
            print(reply)
        return

    # 无参数：交互模式
    print("MiniWorldAgent CLI 交互模式。exit 退出。")
    while True:
        try:
            q = input("\n老板: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if q.lower() in ("exit", "quit", "退出"):
            print("小WM: 再见。")
            break
        print(f"小WM: {bot.reply(q)}")


if __name__ == "__main__":
    main()
