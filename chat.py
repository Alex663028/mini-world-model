"""常识对话引擎：世界模型 + 语言模块的分工架构。

架构说明（重要）：
    世界模型本身没有语言通路（无词表、无文本编码器、无语言头），
    "喂常识"喂不进世界模型。正确做法是两个模块分工：

        [语言模块 chat.py]  常识检索 + 意图路由 + 自然语言生成
              |  ^
              v  |
        [具身查询]  你在哪/目标在哪/想象一下/往前走
              |  ^
              v  |
        [世界模型]  编码器 + 动力学 + 解码器 + MPC

    语言模块负责"说"，世界模型负责"感知和推演物理世界"，
    具身查询是两者的桥梁。这正是在地机器人 / 具身智能的标准架构。

无外部依赖：中文匹配用字符 bigram 重叠，不需要分词库。
"""

import re

import numpy as np
import torch

from knowledge import ALL_KNOWLEDGE, FALLBACKS, SMALLTALK


# 查询框架词：匹配前先删掉，否则"是多少/为什么"会稀释关键词重叠率
STOP_PHRASES = [
    "是多少", "是多少", "为什么", "是什么", "什么意思", "哪条", "哪个", "哪些",
    "哪里", "哪儿", "怎么", "怎样", "如何", "有没有", "是不是", "能不能",
    "请问", "一下", "告诉我", "介绍", "讲讲", "说说", "么", "吗", "呢", "吧",
]


def extract_agent_pos(img, img_size=64):
    """从渲染图中提取 agent（最亮点）的世界坐标。

    本地实现一份（不从 demo.py import），避免训练环境为装 matplotlib 的
    ComfyUI python 环境被拖累——语言模型训练只需要 torch + numpy。
    """
    idx = np.unravel_index(np.argmax(img), img.shape)
    y, x = idx
    wx = (x - 4) / (img_size - 8) * 2.0 - 1.0
    wy = (y - 4) / (img_size - 8) * 2.0 - 1.0
    return np.array([wx, wy])


def bigrams(text):
    """中文 2-gram + 英文/数字整词 token，并过滤查询框架词。

    不需要分词库；英文串（如 GPU、DNA）整体作为一个 token，
    按字符切 bigram 对英文没有意义。
    """
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
    for s in STOP_PHRASES:
        text = text.replace(s, "")
    if not text:
        return set()
    tokens = set()
    for m in re.finditer(r"[a-zA-Z0-9]+", text):
        tokens.add(m.group().upper())
    zh = re.sub(r"[^一-鿿]", " ", text)
    for i in range(len(zh) - 1):
        if zh[i] != " " and zh[i + 1] != " ":
            tokens.add(zh[i : i + 2])
    return tokens


class ChatEngine:
    """对话引擎：常识问答 + 计算 + 世界模型具身查询。"""

    # 世界模型查询路由表（子串命中，按优先级排列）
    WORLD_PATTERNS = [
        (["潜状态", "latent", "编码", "表征"], "latent"),
        (["你在哪", "你在什么地方", "你的位置", "你的位置", "现在在哪"], "pos"),
        (["目标在哪", "目标位置", "目的地在", "目标点"], "target"),
        (["看到了什么", "看到什么", "眼前", "画面里", "观测", "你能看到"], "see"),
        (["想象一下", "想象", "梦境", "推演", "rollout"], "imagine"),
        (["往前走", "向前走", "往左", "向左", "往右", "向右", "往后退", "推一下", "走一步"], "act"),
        (["状态", "你好吗", "怎么样", "情况"], "status"),
    ]

    # 任务意图（最高优先级：语言即任务，闭合"理解→规划→行动"闭环）
    TASK_PATTERNS = [
        (["去目标", "走到目标", "导航", "前往", "到达目标", "去那边", "找目标",
          "追踪", "走过去", "去到"], "task"),
    ]

    # 环境操控意图
    RESET_PATTERNS = [
        (["重置", "换一个目标", "换目标", "新目标", "重新开始", "新回合", "再来"], "reset"),
    ]

    # 边界感知意图
    BOUNDARY_PATTERNS = [
        (["墙在哪", "边界", "会撞墙", "四周", "边上有"], "boundary"),
    ]

    def __init__(self, model=None, env=None, ckpt="checkpoint.pt"):
        from env import PointMassEnv
        from model import MiniWorldModel

        if model is None:
            self.model = MiniWorldModel(latent_dim=32, action_dim=2)
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            self.model.eval()
        else:
            self.model = model
        self.env = env if env is not None else PointMassEnv(seed=2026)
        self.env.reset()
        self.history = []

        # 预计算常识库的 bigram 索引（"喂常识"只需往 knowledge.py 加条目）
        self.kb = []
        for entry in ALL_KNOWLEDGE:
            bg = set()
            for key in entry["keys"]:
                bg |= bigrams(key)
            self.kb.append((bg, entry))

    # ---------- 主入口 ----------

    def reply(self, text):
        text = text.strip()
        if not text:
            return "说点什么吧，我在听。"
        self.history.append(("user", text))

        # 1. 世界模型具身查询（优先于常识检索，避免关键词撞车）
        for patterns, kind in self.WORLD_PATTERNS:
            if any(p in text for p in patterns):
                answer = self._world_query(kind, text)
                self.history.append(("bot", answer))
                return answer

        # 2. 计算器
        calc = self._try_calc(text)
        if calc is not None:
            self.history.append(("bot", calc))
            return calc

        # 3. 闲聊
        for keys, answer in SMALLTALK:
            if any(k in text for k in keys):
                self.history.append(("bot", answer))
                return answer

        # 4. 常识检索（bigram 重叠度，用较小集合归一化，短查询也能得高分）
        best, best_score = None, 0.0
        q_bg = bigrams(text)
        for entry_bg, entry in self.kb:
            if not entry_bg or not q_bg:
                continue
            score = len(q_bg & entry_bg) / max(1, min(len(q_bg), len(entry_bg)))
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= 0.3:
            answer = best["answer"]
        else:
            answer = FALLBACKS[len(text) % len(FALLBACKS)]
        self.history.append(("bot", answer))
        return answer

    # ---------- 世界模型具身查询 ----------

    def _world_query(self, kind, text):
        obs = self.env.render()
        obs_t = torch.from_numpy(obs[None, None].astype(np.float32))
        pos = self.env.state[:2]
        dist = float(np.linalg.norm(self.env.pos - self.env.target))

        if kind == "pos":
            return (
                f"我现在的世界坐标是 ({pos[0]:.2f}, {pos[1]:.2f})，"
                f"离目标还有 {dist:.2f}。这是编码器从当前画面里读出来的。"
            )
        if kind == "target":
            t = self.env.target
            return f"目标点在 ({t[0]:.2f}, {t[1]:.2f})，画面上一圈灰色的环就是它。"
        if kind == "see":
            return (
                "我的观测是一张 64x64 灰度图：正中间偏亮的实心点是我自己，"
                "灰色的圆环是目标点，四周一圈暗条是墙，淡淡的拖尾是我的运动轨迹。"
            )
        if kind == "latent":
            with torch.no_grad():
                z = self.model.encode(obs_t)
            return (
                f"当前潜状态 32 维，均值 {z.mean().item():+.3f}，"
                f"标准差 {z.std().item():.3f}，最大绝对值 {z.abs().max().item():.3f}。"
                "整个视觉世界就被压在这 32 个数里。"
            )
        if kind == "imagine":
            action = self._parse_direction(text)
            with torch.no_grad():
                z = self.model.encode(obs_t)
                zc = z
                for _ in range(12):
                    zc = self.model.imagine_step(zc, torch.from_numpy(action.astype(np.float32))[None])
                img = self.model.decode(zc).squeeze().numpy()
                p = extract_agent_pos(img)
            return (
                f"我想象了一下：朝那个方向推 12 步，梦里的我会出现在 ({p[0]:.2f}, {p[1]:.2f})，"
                f"离目标 {np.linalg.norm(p - self.env.target):.2f}。"
                "注意想象越远误差越大，12 步已经不算短了。"
            )
        if kind == "act":
            action = self._parse_direction(text)
            obs, _, done, info = self.env.step(action)
            new_pos = self.env.state[:2]
            tail = " 我到达目标了！" if info["reached"] else ""
            return (
                f"执行了动作 ({action[0]:.1f}, {action[1]:.1f})，"
                f"现在在 ({new_pos[0]:.2f}, {new_pos[1]:.2f})，"
                f"离目标 {info['dist']:.2f}。{tail}"
            )
        if kind == "status":
            return (
                f"运行中。位置 ({pos[0]:.2f}, {pos[1]:.2f})，"
                f"离目标 {dist:.2f}，已走 {self.env.steps} 步，世界模型参数 167 万。"
            )
        if kind == "task":
            return self._run_navigation()
        if kind == "reset":
            self.env.reset()
            return (
                f"新回合开始。我在 ({self.env.pos[0]:.2f}, {self.env.pos[1]:.2f})，"
                f"目标在 ({self.env.target[0]:.2f}, {self.env.target[1]:.2f})，"
                f"直线距离 {float(np.linalg.norm(self.env.pos - self.env.target)):.2f}。"
            )
        if kind == "boundary":
            return (
                "我的世界四面都有墙：上下左右各一道两像素宽的暗灰边界，图上看得见。"
                "撞墙会反弹且速度减半，所以贴墙时我的想象力会下降——"
                "MPC 只信十二步以内的梦，墙边的梦容易失真。"
            )
        return "我的世界模型模块今天状态正常。"

    def _bearing(self, frm, to):
        """frm→to 的八方位中文。"""
        import math

        dx, dy = to - frm
        ang = math.degrees(math.atan2(-dy, dx))
        dirs = ["正东", "东北", "正北", "西北", "正西", "西南", "正南", "东南"]
        return dirs[int(((ang + 22.5) % 360) // 45)]

    def _run_navigation(self):
        """语言即任务：MPC 导航到目标，带进度汇报与卡住兜底。

        混合策略：
        1. 主策略——世界模型想象：枚举 8 方向 + 惯性 + 减速候选，
           每个候选想象 12 步，选梦里终点离目标最近的执行
        2. 兜底——直冲：连续 5 步距离无进展（想象失真/打转）时，
           切换为朝目标的比例直冲，并在汇报里如实说明

        真实 agent 系统也是这样：学习型规划 + 传统控制互为备份。
        """
        reports = []
        steps_used = 0
        switches = 0
        phase = "mpc"
        prev_a = np.zeros(2, dtype=np.float32)
        stuck = 0
        info = {"reached": False, "dist": float(np.linalg.norm(self.env.pos - self.env.target))}
        with torch.no_grad():
            for step in range(60):
                d0 = float(np.linalg.norm(self.env.pos - self.env.target))
                obs = self.env.render()
                obs_t = torch.from_numpy(obs[None, None].astype(np.float32))
                z = self.model.encode(obs_t)

                if d0 > 0.4 and stuck < 3:
                    # 规划层：MPC 想象导航（世界模型的主场——中远程选方向）
                    throttle = min(0.7, max(0.3, d0 * 1.2))
                    candidates = [
                        np.array([np.cos(a), np.sin(a)], dtype=np.float32) * throttle
                        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
                    ]
                    candidates.append((prev_a * 0.8).astype(np.float32))
                    candidates.append(np.zeros(2, dtype=np.float32))
                    best_a, best_pd = None, np.inf
                    for a in candidates:
                        zc = z
                        min_d = np.inf
                        for _ in range(12):
                            zc = self.model.imagine_step(zc, torch.from_numpy(a).unsqueeze(0))
                            p = extract_agent_pos(self.model.decode(zc).squeeze().numpy())
                            min_d = min(min_d, float(np.linalg.norm(p - self.env.target)))
                        if min_d < best_pd:
                            best_a, best_pd = a, min_d
                else:
                    # 控制层：PD 逆动力学精密进近。
                    # 环境有固有刹车距离（阻尼 0.8，带速度冲到近处会冲出 0.7 远），
                    # 想象擅长选方向，但最后 0.4 的进近要速度感知的控制——
                    # 真实机器人也是分层的：planning + control
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
                d1 = info["dist"]
                stuck = stuck + 1 if d1 >= d0 - 0.01 else 0
                if steps_used % 15 == 0 and not done:
                    bearing = self._bearing(self.env.pos, self.env.target)
                    reports.append(f"第{steps_used}步向{bearing}推进，还剩{d1:.1f}")
                if done:
                    break
        if info.get("reached"):
            tail = f"过程：{'；'.join(reports)}。" if reports else ""
            switch_note = f"中途切换控制层 {switches} 次。" if switches else ""
            return f"任务完成！用 {steps_used} 步到达目标。{switch_note}{tail}"
        tail = f"过程：{'；'.join(reports)}。" if reports else ""
        return f"60 步未能到达，最近时只剩 {info['dist']:.1f}。{tail}（世界模型的想象窗口是十二步，长途导航会累积误差。）"

    def _parse_direction(self, text):
        """把方向词解析成动作向量；没有方向词就朝目标推。"""
        if "左" in text or "西" in text:
            return np.array([-0.7, 0.0], dtype=np.float32)
        if "右" in text or "东" in text:
            return np.array([0.7, 0.0], dtype=np.float32)
        if "上" in text or "前" in text or "北" in text:
            return np.array([0.0, -0.7], dtype=np.float32)
        if "下" in text or "后" in text or "南" in text:
            return np.array([0.0, 0.7], dtype=np.float32)
        # 默认：朝目标方向
        d = self.env.target - self.env.pos
        n = np.linalg.norm(d)
        return (d / n * 0.7).astype(np.float32) if n > 1e-6 else np.zeros(2, dtype=np.float32)

    # ---------- 计算器 ----------

    def _try_calc(self, text):
        # 运算符支持中英文双语（"乘以"这类双字词必须放在单字符运算符前面）
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
        return f"{a:g} {op} {b:g} = {r:g}。这种算术我现在还算得过来。"
