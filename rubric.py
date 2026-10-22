"""Quality criteria shared by the Opus judge (judge_cli.py) and Opus scoring (score_cli.py)."""
import hashlib

CRITERIA = """评估对象：一个“只用 emoji 回复”的聊天机器人，每条回复是 1 到 3 个 emoji。

标准（按重要性）：
1. 贴切：像朋友那样回应了这条消息的内容和情绪，而不是答非所问，也不是把用户的话逐字翻译成 emoji。
2. 情绪得当：对难过、生气或敏感的话题，回应要合适、有温度，不轻佻。
3. 精准：每个 emoji 的意思都对得上，没有莫名其妙或错误的 emoji。

注意：
- 1 个、2 个、3 个 emoji 都是正常的回复长度，只看贴不贴切，不要因为 emoji 更多或更少就偏向某一方。一个恰到好处的 emoji 可以比三个更好。
- 重复同一个 emoji（如 😂😂）是正常的加强语气方式，不扣分。
- 放在任何消息后面都说得通的万能 emoji（如 🤔、👍、😊）如果没有真正回应这条消息的内容，要明显低分，重复三次也一样。"""


def rubric_hash(*texts: str) -> str:
    """Short hash so cached Opus results are invalidated when the rubric changes."""
    return hashlib.sha1("".join(texts).encode()).hexdigest()[:6]
