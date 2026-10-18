"""MiniWorldAgent HTTP API（stdlib 零依赖，端口 8399）。

端点：
    GET  /state     环境状态 + 系统信息（JSON）
    GET  /history   最近对话历史
    POST /reply     {"text": "..."} → {"reply": "...", "state": {...}}
    POST /task      {"task": "navigate"} → 执行任务并汇报
    POST /perceive  → 当前感知（潜状态 + 文字描述）
    POST /imagine   {"direction": "前"} → 世界模型想象
    POST /act       {"direction": "左"} → 单步动作
    POST /reset     → 重置环境
    POST /kb        {"question": "..."} → 常识查询
    POST /calc      {"expression": "23 乘以 7"} → 算术

用法：
    python api_server.py [--port 8399]

安全/并发说明（默认值都按"本机开发"设，别直接暴露公网）：
    - 默认只绑 127.0.0.1；显式 --host 0.0.0.0 时建议同时设 --api-key。
    - 设了 --api-key（或环境变量 MINIWORLD_API_KEY）后，所有请求必须带
      `Authorization: Bearer <key>` 或 `X-API-Key: <key>`，否则 401。
    - 请求体上限 64KB，超了 413；非 JSON / 非对象体 → 400（不再静默当空 body）。
    - 用 ThreadingHTTPServer + 全局 RLock：env 是单实例有状态对象，
      navigation 会跑几十秒，此时其它请求排队而不是读到半截状态。

调用示例：
    curl -X POST http://127.0.0.1:8399/reply -H "Content-Type: application/json" -d "{\"text\": \"你看到了什么？\"}"
"""

import argparse
import hmac
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from agent import MiniWorldAgent

BOT = None  # lazy 初始化（首次请求时加载模型）
# 一个 agent 实例内部是有状态的 env / history；ThreadingHTTPServer 下必须串行化。
# RLock 是为了让 get_bot() 的懒加载和请求处理共用一把锁而不自锁。
BOT_LOCK = threading.RLock()
MAX_BODY = 64 * 1024      # 请求体上限
API_KEY = ""              # 非空则开启鉴权（--api-key 或 MINIWORLD_API_KEY）


class BadRequest(Exception):
    """请求体格式错误 → 400。"""


class PayloadTooLarge(Exception):
    """请求体超限 → 413。"""


def get_bot():
    global BOT
    with BOT_LOCK:
        if BOT is None:
            BOT = MiniWorldAgent()
        return BOT


def snapshot():
    bot = get_bot()
    return {
        "agent_pos": [round(float(v), 3) for v in bot.env.pos],
        "target_pos": [round(float(v), 3) for v in bot.env.target],
        "distance": round(float(np.linalg.norm(bot.env.pos - bot.env.target)), 3),
        "steps": bot.env.steps,
        "lm_loaded": bot.lm is not None,
        "state_dim": bot.state_dim,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        """设了 API_KEY 就校验请求头；用 compare_digest 避免时序侧信道。"""
        if not API_KEY:
            return True
        provided = self.headers.get("X-API-Key", "") or ""
        auth = self.headers.get("Authorization", "") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        return bool(provided) and hmac.compare_digest(provided, API_KEY)

    def _body(self):
        """解析 JSON 请求体：报错就抛，绝不静默 return {}（那会变成 200 空操作）。"""
        length_raw = self.headers.get("Content-Length")
        if length_raw is None:
            raise BadRequest("缺少 Content-Length 头")
        try:
            length = int(length_raw)
        except ValueError:
            raise BadRequest(f"Content-Length 不是整数：{length_raw!r}")
        if length < 0:
            raise BadRequest("Content-Length 不能为负")
        if length > MAX_BODY:
            raise PayloadTooLarge(f"请求体 {length} 字节超过上限 {MAX_BODY} 字节")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise BadRequest(f"请求体不是合法 JSON：{e}")
        if not isinstance(data, dict):
            raise BadRequest("请求体必须是 JSON 对象")
        return data

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s\n" % (fmt % args))

    def do_GET(self):
        if not self._authorized():
            self._send({"ok": False, "error": "unauthorized"}, 401)
            return
        with BOT_LOCK:
            bot = get_bot()
            if self.path == "/state":
                self._send({"ok": True, "state": snapshot()})
            elif self.path == "/history":
                self._send({"ok": True, "history": bot.history[-10:]})
            elif self.path == "/":
                self._send({
                    "ok": True,
                    "service": "MiniWorldAgent API",
                    "endpoints": {
                        "GET  /state": "环境与事件状态",
                        "GET  /history": "最近 10 轮对话",
                        "POST /reply": '{"text": "你好"} — 自由对话（路由模式）',
                        "POST /task": '{"task": "去目标那儿"} — 执行导航任务',
                        "POST /perceive": '{"question": "你看到了什么？"} — 感知问答',
                        "POST /imagine": '{"direction": "左"} — 想象一步',
                        "POST /act": '{"direction": "前"} — 执行动作',
                        "POST /reset": "重置环境",
                        "POST /kb": '{"question": "长城是什么"} — 常识检索',
                        "POST /calc": '{"expression": "23*7"} — 四则运算',
                    },
                })
            else:
                self._send({"ok": False, "error": f"unknown path {self.path}"}, 404)

    def do_POST(self):
        if not self._authorized():
            self._send({"ok": False, "error": "unauthorized"}, 401)
            return
        try:
            data = self._body()
        except PayloadTooLarge as e:
            self._send({"ok": False, "error": str(e)}, 413)
            return
        except BadRequest as e:
            self._send({"ok": False, "error": str(e)}, 400)
            return
        path = self.path
        try:
            with BOT_LOCK:  # env/history 是共享单实例状态，请求串行化
                bot = get_bot()
                if path == "/reply":
                    text = data.get("text", "")
                    reply = bot.reply(text)
                    self._send({"ok": True, "input": text, "reply": reply, "state": snapshot()})
                elif path == "/task":
                    task = data.get("task", "去目标那儿")
                    reply = bot.reply(task)
                    self._send({"ok": True, "task": task, "reply": reply, "state": snapshot()})
                elif path == "/perceive":
                    reply = bot.reply("你看到了什么？")
                    self._send({"ok": True, "reply": reply, "state": snapshot()})
                elif path == "/imagine":
                    direction = data.get("direction", "前")
                    reply = bot.reply(f"想象一下往{direction}走")
                    self._send({"ok": True, "direction": direction, "reply": reply})
                elif path == "/act":
                    direction = data.get("direction", "前")
                    reply = bot.reply(f"往{direction}走")
                    self._send({"ok": True, "direction": direction, "reply": reply, "state": snapshot()})
                elif path == "/reset":
                    reply = bot.reply("重置")
                    self._send({"ok": True, "reply": reply, "state": snapshot()})
                elif path == "/kb":
                    q = data.get("question", "")
                    hit = bot._retrieve(q)
                    self._send({"ok": True, "question": q,
                                "answer": hit[0] if hit else None,
                                "score": round(hit[1], 3) if hit else 0})
                elif path == "/calc":
                    expr = data.get("expression", "")
                    reply = bot._try_calc(expr) or "无法解析这个算式"
                    self._send({"ok": True, "expression": expr, "result": reply})
                else:
                    self._send({"ok": False, "error": f"unknown path {path}"}, 404)
        except Exception as e:
            # HTTP 边界兜底：单个请求出错不能让服务器线程带着 traceback 死掉，
            # 但 traceback 必须落日志（否则 500 排查不了）
            sys.stderr.write("[api] request failed: %s\n" % traceback.format_exc())
            self._send({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def main():
    global API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-key", default=os.environ.get("MINIWORLD_API_KEY", ""),
                        help="非空则要求 Authorization: Bearer <key> / X-API-Key")
    args = parser.parse_args()
    API_KEY = args.api_key
    if args.host not in ("127.0.0.1", "localhost", "::1") and not API_KEY:
        print(f"[warn] 绑定到 {args.host} 且未设 --api-key：接口无鉴权，"
              f"请勿暴露到公网（或用 --api-key / MINIWORLD_API_KEY 开启鉴权）",
              file=sys.stderr)
    get_bot()  # 启动时预热（加载模型）
    # ThreadingHTTPServer：并发请求进不同线程（配合 BOT_LOCK 串行访问 agent）
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MiniWorldAgent API listening on http://{args.host}:{args.port}"
          + ("（已开启 API key 鉴权）" if API_KEY else ""))
    print("endpoints: GET /state /history | POST /reply /task /perceive /imagine /act /reset /kb /calc")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
