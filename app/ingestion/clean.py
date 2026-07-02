"""
Cleans and normalizes the raw scraped SHL catalog into a consistent schema.

Input:  app/catalog/raw_catalog.json  (377 records, as scraped)
Output: app/catalog/cleaned_catalog.json

What this does and why:
- Derives a `test_type` letter-code list from the `keys` category list, because
  the raw scrape has no letter-code field, but every conversation trace we were
  given (and the assignment's own example response) displays test types as
  single letters (K, P, A, S, B, C, D). We reverse-engineer the mapping from the
  traces themselves rather than guessing.
- Normalizes `duration` into a human string (kept as scraped, e.g. "18 minutes")
  and a numeric `duration_minutes` (int or None) for filtering/sorting. Missing
  duration becomes "Variable" only when duration_raw hints at variability,
  otherwise "Not specified" -- we don't silently invent "Variable".
- Leaves `name` untouched. The traces show official names keep suffixes like
  "(New)" -- SHL's own catalog treats that as part of the product name, so
  stripping it would make our recommendations mismatch the ground truth.
- Builds a single `search_text` field per item (name + description + keys +
  job_levels) for embedding. This is what retrieval.py embeds -- keeping it
  separate from the display fields means we can tune retrieval text without
  touching what gets shown to the user.
"""
import json
import re
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "catalog" / "raw_catalog.json"
OUT_PATH = Path(__file__).parent.parent / "catalog" / "cleaned_catalog.json"

# Mapping derived from the 10 provided conversation traces + assignment doc.
# Every trace table's "Test Type" column uses these single-letter codes.
KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def parse_duration(duration: str, duration_raw: str):
    duration = (duration or "").strip()
    duration_raw = (duration_raw or "").strip()

    if duration:
        m = re.search(r"(\d+)", duration)
        minutes = int(m.group(1)) if m else None
        return duration, minutes

    if "variable" in duration_raw.lower():
        return "Variable", None

    if "max" in duration_raw.lower():
        m = re.search(r"(\d+)", duration_raw)
        if m:
            return f"Up to {m.group(1)} minutes", int(m.group(1))

    return "Not specified", None


def clean_record(item: dict) -> dict:
    keys = item.get("keys", [])
    test_type = [KEY_TO_CODE.get(k, "") for k in keys]
    test_type = [t for t in test_type if t]  # drop unmapped, keep order

    duration_display, duration_minutes = parse_duration(
        item.get("duration", ""), item.get("duration_raw", "")
    )

    languages = item.get("languages", []) or []
    job_levels = item.get("job_levels", []) or []
    description = (item.get("description") or "").strip()

    search_text = " | ".join(
        filter(
            None,
            [
                item["name"],
                description,
                ", ".join(keys),
                ", ".join(job_levels),
            ],
        )
    )

    return {
        "id": item["entity_id"],
        "name": item["name"],
        "url": item["link"],
        "test_type": test_type,
        "keys": keys,
        "duration_display": duration_display,
        "duration_minutes": duration_minutes,
        "languages": languages,
        "job_levels": job_levels,
        "remote": item.get("remote") == "yes",
        "adaptive": item.get("adaptive") == "yes",
        "description": description,
        "search_text": search_text,
    }


def main():
    raw = json.loads(RAW_PATH.read_text(), strict=False)
    cleaned = [clean_record(item) for item in raw]

    # sanity checks -- fail loudly rather than silently shipping bad data
    assert len(cleaned) == len(raw), "record count changed during cleaning"
    ids = [c["id"] for c in cleaned]
    assert len(ids) == len(set(ids)), "duplicate ids after cleaning"
    no_type = [c["name"] for c in cleaned if not c["test_type"]]
    if no_type:
        print(f"WARNING: {len(no_type)} records have no mapped test_type: {no_type[:5]}")

    OUT_PATH.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    print(f"Cleaned {len(cleaned)} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
