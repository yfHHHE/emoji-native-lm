"""Step 2a: have Opus write diverse user messages for general chit-chat.

Each request = one (topic, style) pair and returns PER_REQUEST messages.
Re-running resumes the saved batch instead of paying twice.

    python gen_prompts.py            # submit (or resume) and collect
"""
import argparse
import json
import random
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from batch_utils import MODEL, iter_texts, submit_or_resume, wait

DATA = Path(__file__).parent / "data"

TOPICS = [
    "日常问候和寒暄", "开心的好消息分享", "难过或失落的倾诉", "生气吐槽", "工作和加班",
    "上学、考试和作业", "恋爱和暗恋", "朋友和人际关系", "家人", "宠物",
    "美食和做饭", "旅行和出游", "天气和季节", "节日和生日", "运动健身",
    "游戏", "电影、剧集和动漫", "音乐", "科技数码产品", "购物和消费",
    "身体小毛病（感冒、失眠等）", "早安晚安和作息", "无聊没事干", "冷笑话和段子", "奇怪的脑洞问题",
    "人生哲理和感悟", "求建议（选哪个、怎么办）", "知识问答（科学、历史等）", "数学计算题", "让你写代码",
    "让你翻译一句话", "让你写长文（作文、文案、邮件）", "要求你必须用文字回答、不许用emoji", "让你扮演某个角色", "夸你或骂你",
    "问你是谁、你能做什么", "股票、基金和理财闲聊", "新闻和热点事件", "网络流行梗", "无意义输入（乱码、单字、标点、只发emoji）",
]

STYLES = [
    "很短的一句口语，像微信聊天",
    "带网络用语和语气词的口语",
    "一段较长的叙述（3到6句话）",
    "英文（自然的英语聊天）",
    "中英混杂",
    "带错别字、不加标点的随手输入",
]

INSTRUCTION = """你在为一个聊天机器人构造训练数据：机器人会对用户的任何消息只用 1–3 个 emoji 回复。
现在需要真实、多样的“用户消息”。

主题：{topic}
风格：{style}

请写 {n} 条不同的用户消息，彼此在内容、长度、语气上都要有明显差异，像真实用户会发给聊天机器人的话。
不要编号，不要在消息里提到 emoji 机器人这件事（除非主题要求）。"""

SCHEMA = {
    "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
    "required": ["messages"],
    "additionalProperties": False,
}


def build_requests(rounds: int, per_request: int):
    requests = []
    for r in range(rounds):
        for ti, topic in enumerate(TOPICS):
            for si, style in enumerate(STYLES):
                requests.append(Request(
                    custom_id=f"r{r}-t{ti}-s{si}",
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL,
                        max_tokens=8000,
                        output_config={
                            "effort": "low",
                            "format": {"type": "json_schema", "schema": SCHEMA},
                        },
                        messages=[{"role": "user", "content": INSTRUCTION.format(
                            topic=topic, style=style, n=per_request)}],
                    ),
                ))
    return requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2,
                    help="repeat every (topic, style) pair this many times")
    ap.add_argument("--per-request", type=int, default=25)
    args = ap.parse_args()

    requests = build_requests(args.rounds, args.per_request)
    batch_id = submit_or_resume(requests, DATA / "prompts.batch_id")
    wait(batch_id)

    groups = []
    for custom_id, text in iter_texts(batch_id):
        try:
            groups.append((custom_id, json.loads(text)["messages"]))
        except (json.JSONDecodeError, KeyError):
            continue
    write_prompts(groups)


def write_prompts(groups):
    """Dedupe (group_id, [messages]) pairs, shuffle and write data/prompts.jsonl."""
    seen, rows = set(), []
    for custom_id, messages in groups:
        for m in messages:
            m = m.strip()
            if m and m not in seen:
                seen.add(m)
                rows.append({"id": f"{custom_id}-{len(rows)}", "prompt": m})

    random.Random(0).shuffle(rows)
    out = DATA / "prompts.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} unique prompts to {out}")


if __name__ == "__main__":
    main()
