"""Step 1: output spec. A valid reply is 1..MAX_EMOJI emoji and nothing else."""
import emoji

MIN_EMOJI = 1
MAX_EMOJI = 3


def normalize(text: str) -> str:
    """Drop all whitespace; model replies should be emoji glued together."""
    return "".join(text.split())


def parse_emojis(text: str) -> list[str] | None:
    """Return the emoji in `text` if it consists only of emoji, else None.

    Uses the `emoji` package, so ZWJ sequences (👨‍👩‍👧), skin tones and flags
    count as one emoji each.
    """
    s = normalize(text)
    if not s:
        return None
    found = emoji.emoji_list(s)
    if "".join(e["emoji"] for e in found) != s:
        return None
    return [e["emoji"] for e in found]


def is_valid_reply(text: str) -> bool:
    found = parse_emojis(text)
    return found is not None and MIN_EMOJI <= len(found) <= MAX_EMOJI


def finalize_reply(text: str) -> str:
    """Post-processing applied to every model reply before it is shown:
    a single emoji is repeated MAX_EMOJI times (😂 -> 😂😂😂); 2-3 emoji pass through."""
    found = parse_emojis(text)
    if found is not None and len(found) == 1:
        return found[0] * MAX_EMOJI
    return normalize(text)


if __name__ == "__main__":
    cases = {
        "😂": True,
        "😂👍": True,
        "🎉🎂🥳": True,
        "🎉 🎂 🥳": True,
        "👨‍👩‍👧🏠": True,
        "👍🏽": True,
        "🇨🇳": True,
        "😂👍🎉🔥": False,
        "": False,
        "好的😂": False,
        "😂!": False,
        "<think></think>😂": False,
    }
    for text, want in cases.items():
        got = is_valid_reply(text)
        print(("OK  " if got == want else "FAIL"), repr(text), got)
    assert all(is_valid_reply(t) == w for t, w in cases.items())

    finals = {"😂": "😂😂😂", "👨‍👩‍👧": "👨‍👩‍👧👨‍👩‍👧👨‍👩‍👧", "🇨🇳": "🇨🇳🇨🇳🇨🇳",
              "😂👍": "😂👍", "🎉 🎂 🥳": "🎉🎂🥳", "好的": "好的"}
    for text, want in finals.items():
        got = finalize_reply(text)
        print(("OK  " if got == want else "FAIL"), repr(text), "->", got)
    assert all(finalize_reply(t) == w for t, w in finals.items())
    print("all passed")
