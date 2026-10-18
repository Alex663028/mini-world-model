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

调用示例：
    curl -X POST http://127.0.0.1:8399/reply -H "Content-Type: application/json" -d "{\"text\": \"你看到了什么？\"}"
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from agent import MiniWorldAgent

BOT = None  # lazy 初始化（首次请求时加载模型）


def get_bot():
    global BOT
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

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] %s\n" % (fmt % args))

    def do_GET(self):
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
        bot = get_bot()
        data = self._body()
        path = self.path
        try:
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
            self._send({"ok": False, "error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    get_bot()  # 启动时预热（加载模型）
    server = HTTPServer((args.host, args.port), Handler)
    print(f"MiniWorldAgent API listening on http://{args.host}:{args.port}")
    print("endpoints: GET /state /history | POST /reply /task /perceive /imagine /act /reset /kb /calc")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
