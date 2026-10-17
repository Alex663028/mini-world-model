"""大语料汇总：问答对 + 段落式连续文本。

- 问答对：每个知识点生成多种问法（教模型"听懂问题并作答"）
- 段落模式：把同主题陈述句串成连续文本（教模型行文连贯，不只是问答复读机）

输出 corpus_big.txt。
"""

from knowledge import ALL_KNOWLEDGE as KB_SMALL
from science_big import SCIENCE_BIG
from history_geo_big import HISTORY_BIG
from tech_big import TECH_BIG
from life_big import LIFE_BIG

ALL_BIG = SCIENCE_BIG + HISTORY_BIG + TECH_BIG + LIFE_BIG

TEMPLATES = [
    "什么是{}？",
    "{}是什么？",
    "讲讲{}",
    "{}指什么？",
]


def build():
    lines = []

    # 1. 大语料问答对（每条 key 生成全部问法变体）
    for key, fact in ALL_BIG:
        for tpl in TEMPLATES:
            lines.append(f"问：{tpl.format(key)}\n答：{fact}\n")

    # 2. 原有常识库问答对（保留，和小语料模型可对比）
    for entry in KB_SMALL:
        for key in entry["keys"]:
            lines.append(f"问：什么是{key}？\n答：{entry['answer']}\n")

    # 3. 世界模型自述问答（保留项目自身的事实）
    from corpus import GENERAL_DIALOG, SELF_DIALOG

    for q, a in GENERAL_DIALOG:
        lines.append(f"问：{q}\n答：{a}\n")
    for q, a in SELF_DIALOG:
        lines.append(f"问：{q}\n答：{a}\n")

    # 4. 段落式连续文本（按主题把陈述句串起来，学行文连贯性）
    for name, group in [
        ("自然科学", SCIENCE_BIG),
        ("历史地理", HISTORY_BIG),
        ("技术工程", TECH_BIG),
        ("文化生活", LIFE_BIG),
    ]:
        facts = [f for _, f in group]
        # 每 8 条串一段，段首加主题引导
        for i in range(0, len(facts), 8):
            chunk = facts[i : i + 8]
            para = f"下面聊{name}。" + "".join(chunk)
            lines.append(f"{para}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    text = build()
    with open("corpus_big.txt", "w", encoding="utf-8") as f:
        f.write(text)
    n_qa = sum(1 for line in text.split("\n") if line.startswith("问："))
    print(f"total chars: {len(text)}")
    print(f"QA pairs: {n_qa}")
    print(f"unique chars: {len(set(text))}")
