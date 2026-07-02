"""
Parses the provided .md conversation traces into structured turns:
[{"user": str, "expected_names": [str, ...] | None, "end_of_conversation": bool}, ...]

expected_names is None on turns with no recommendation table (clarify/compare
turns) and a list of assessment names (possibly empty) on turns that carry a
shortlist table -- this is what we diff our own output against for recall.
"""
import re
from pathlib import Path

USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
EOC_RE = re.compile(r"`end_of_conversation`:\s*\*\*(true|false)\*\*")
TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|", re.MULTILINE)
TURN_SPLIT_RE = re.compile(r"### Turn \d+")


def parse_trace(path: Path):
    text = path.read_text()
    turn_blocks = TURN_SPLIT_RE.split(text)[1:]  # drop preamble before first "### Turn"
    turns = []
    for block in turn_blocks:
        user_match = USER_BLOCK_RE.search(block)
        if not user_match:
            continue
        user_text = user_match.group(1).strip()

        # everything after "**Agent**" is the agent section for this turn
        agent_section = block.split("**Agent**", 1)[1] if "**Agent**" in block else ""

        has_table = bool(re.search(r"\|\s*#\s*\|\s*Name\s*\|", agent_section))
        expected_names = None
        if has_table:
            rows = TABLE_ROW_RE.findall(agent_section.split("|---", 1)[-1])
            expected_names = [r.strip() for r in rows]

        eoc_match = EOC_RE.search(agent_section)
        end_of_conversation = eoc_match.group(1) == "true" if eoc_match else False

        turns.append(
            {
                "user": user_text,
                "expected_names": expected_names,
                "end_of_conversation": end_of_conversation,
            }
        )
    return turns


if __name__ == "__main__":
    import sys
    import json

    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "traces" / "C1.md"
    for t in parse_trace(p):
        print(json.dumps(t, indent=2))
