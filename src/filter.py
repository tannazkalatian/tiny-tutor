"""
Self-critique filtering: score each generated example and keep only those scoring >= min_score.

Usage:
    python src/filter.py \
        --input data/raw/generated.jsonl \
        --output data/filtered.jsonl \
        --min-score 4.0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic

RATE_TOOL = {
    "name": "rate_explanation",
    "description": "Rate an ELI5 explanation on three criteria.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clarity": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Would a 7-year-old follow this? 5 = completely. 1 = lost after sentence 1.",
            },
            "accuracy": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Factually correct? Simplification is fine; distortion is not.",
            },
            "age_appropriateness": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Right tone, no unexplained jargon, no condescension.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the lowest score given.",
            },
        },
        "required": ["clarity", "accuracy", "age_appropriateness", "reasoning"],
    },
}

RATE_SYSTEM = (
    "You are a strict quality reviewer for children's educational content. "
    "A score of 4 means genuinely good — not just okay. Reserve 5 for exceptional clarity."
)


def build_rate_prompt(concept: str, explanation: str) -> str:
    return (
        f'Rate the following ELI5 explanation of "{concept}" on three criteria.\n\n'
        f'Explanation:\n"""\n{explanation}\n"""\n\n'
        "Use the rate_explanation tool."
    )


def score_composite(clarity: int, accuracy: int, age_appropriateness: int) -> float:
    return 0.4 * clarity + 0.4 * accuracy + 0.2 * age_appropriateness


def rate_one(client: anthropic.Anthropic, record: dict, model: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=RATE_SYSTEM,
        tools=[RATE_TOOL],
        tool_choice={"type": "tool", "name": "rate_explanation"},
        messages=[
            {
                "role": "user",
                "content": build_rate_prompt(record["concept"], record["explanation"]),
            }
        ],
    )
    ratings = response.content[0].input
    composite = score_composite(
        ratings["clarity"], ratings["accuracy"], ratings["age_appropriateness"]
    )
    return {
        **record,
        "score": round(composite, 2),
        "score_clarity": ratings["clarity"],
        "score_accuracy": ratings["accuracy"],
        "score_age": ratings["age_appropriateness"],
        "score_reasoning": ratings["reasoning"],
    }


def load_scored(output_path: Path) -> set[str]:
    done = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["concept"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    already_scored = load_scored(output_path)
    print(f"Loaded {len(records)} records. {len(already_scored)} already scored.", flush=True)

    client = anthropic.Anthropic()
    kept = 0
    dropped = 0
    errors = 0

    with output_path.open("a") as out_f:
        for i, record in enumerate(records):
            if record["concept"] in already_scored:
                continue

            try:
                scored = rate_one(client, record, args.model)
            except anthropic.APIError as e:
                print(f"API error on '{record['concept']}': {e}", file=sys.stderr)
                errors += 1
                continue

            if scored["score"] >= args.min_score:
                out_f.write(json.dumps(scored) + "\n")
                out_f.flush()
                kept += 1
            else:
                dropped += 1

            if (i + 1) % 100 == 0:
                total = kept + dropped
                pass_rate = kept / total * 100 if total else 0
                print(
                    f"[{i+1}/{len(records)}] kept={kept} dropped={dropped} "
                    f"pass_rate={pass_rate:.1f}%",
                    flush=True,
                )

            time.sleep(args.delay)

    total = kept + dropped
    pass_rate = kept / total * 100 if total else 0
    print(f"\nDone. kept={kept} dropped={dropped} errors={errors} pass_rate={pass_rate:.1f}%")


if __name__ == "__main__":
    main()
