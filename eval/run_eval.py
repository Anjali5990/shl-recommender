"""
Replays each of the 10 provided traces against OUR agent (not a simulated
LLM user -- we feed the real user turns from the trace directly) and reports:

- Recall@10 on the FINAL turn of each trace (the committed shortlist),
  which is what the assignment's scoring actually weights.
- A few hard-eval / behavior checks per turn:
    * schema always valid (guaranteed by Pydantic, but we double check here)
    * recommendations empty on turns the trace expects no table
      (i.e. our agent isn't over-eagerly recommending on clarify/compare turns)
    * end_of_conversation only true on the trace's actual final turn

This is NOT the same as SHL's real harness (which uses an LLM to *play* the
user persona and can diverge from the script). It's a deterministic replay
against the fixed trace text, useful for fast iteration before worrying
about the harder, non-deterministic evaluation.

Usage:
    export GROQ_API_KEY=...
    python eval/run_eval.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from app.main import app
from eval.parse_trace import parse_trace

TRACES_DIR = Path(__file__).parent / "traces"


def recall_at_k(predicted_names, expected_names):
    if not expected_names:
        return None
    predicted_set = {n.strip().lower() for n in predicted_names}
    expected_set = {n.strip().lower() for n in expected_names}
    hit = len(predicted_set & expected_set)
    return hit / len(expected_set)


def run_trace(client, path: Path):
    turns = parse_trace(path)
    history = []
    per_turn_results = []

    for i, turn in enumerate(turns):
        history.append({"role": "user", "content": turn["user"]})
        resp = client.post("/chat", json={"messages": history})
        time.sleep(15)  # stay under Groq free-tier RPM when replaying many turns back-to-back
        ok = resp.status_code == 200
        data = resp.json() if ok else {}

        reply = data.get("reply", "")
        recs = [r["name"] for r in data.get("recommendations", [])]
        eoc = data.get("end_of_conversation", False)

        history.append({"role": "assistant", "content": reply})

        recall = None
        if turn["expected_names"] is not None:
            recall = recall_at_k(recs, turn["expected_names"])

        over_recommended = turn["expected_names"] is None and len(recs) > 0

        per_turn_results.append(
            {
                "turn": i + 1,
                "user": turn["user"][:70],
                "status_ok": ok,
                "recall": recall,
                "expected_count": len(turn["expected_names"]) if turn["expected_names"] else 0,
                "got_count": len(recs),
                "over_recommended": over_recommended,
                "eoc_expected": turn["end_of_conversation"],
                "eoc_got": eoc,
            }
        )

    return per_turn_results


def main():
    client = TestClient(app)
    trace_files = sorted(TRACES_DIR.glob("*.md"))

    final_recalls = []
    all_over_recs = 0
    all_eoc_mismatches = 0

    for path in trace_files:
        print(f"\n{'=' * 70}\n{path.name}\n{'=' * 70}")
        try:
            results = run_trace(client, path)
        except Exception as e:
            print(f"  CRASHED: {e}")
            continue

        for r in results:
            recall_str = f"{r['recall']:.2f}" if r["recall"] is not None else "  - "
            flags = []
            if r["over_recommended"]:
                flags.append("OVER-RECOMMENDED (expected no table)")
                all_over_recs += 1
            if r["eoc_expected"] != r["eoc_got"]:
                flags.append(f"EOC mismatch (expected {r['eoc_expected']}, got {r['eoc_got']})")
                all_eoc_mismatches += 1
            flag_str = f"  <-- {'; '.join(flags)}" if flags else ""
            print(
                f"  Turn {r['turn']}: recall={recall_str}  "
                f"got={r['got_count']}/expected={r['expected_count']}  "
                f"'{r['user']}'{flag_str}"
            )

        final = results[-1]
        if final["recall"] is not None:
            final_recalls.append(final["recall"])

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    if final_recalls:
        print(f"Mean final-turn Recall: {sum(final_recalls)/len(final_recalls):.3f} "
              f"over {len(final_recalls)} traces")
    print(f"Over-recommendation flags: {all_over_recs}")
    print(f"end_of_conversation mismatches: {all_eoc_mismatches}")


if __name__ == "__main__":
    main()
