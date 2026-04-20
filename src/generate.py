"""
Synthetic ELI5 data generation using the Claude API.

Usage:
    python src/generate.py \
        --concepts data/raw/seed_concepts.jsonl \
        --output data/raw/generated.jsonl \
        --model claude-haiku-4-5-20251001 \
        --max-concepts 2000
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

ANALOGY_STYLES = ["animals", "food", "everyday_objects", "sports", "building"]

SYSTEM_PROMPT = (
    "You are an expert science communicator who specializes in explaining complex STEM concepts "
    "to children aged 5–10. Your explanations are warm, vivid, and always grounded in a concrete "
    "analogy drawn from the child's everyday world. You never use jargon without immediately "
    "replacing it with something simpler."
)

EXPLAIN_TOOL = {
    "name": "explain_concept",
    "description": "Return a structured ELI5 explanation of a STEM concept.",
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "The ELI5 explanation, 3–6 sentences.",
            },
            "analogy_type": {
                "type": "string",
                "enum": ["animals", "food", "everyday_objects", "sports", "building"],
                "description": "The analogy category actually used.",
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "How hard this concept is for a 7-year-old to grasp.",
            },
            "key_analogy": {
                "type": "string",
                "description": "The specific analogy object used (e.g. 'pizza', 'anthill').",
            },
        },
        "required": ["explanation", "analogy_type", "difficulty", "key_analogy"],
    },
}

# Pricing per million tokens (as of early 2025) — update if stale
COST_PER_MTK = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}


def build_user_prompt(concept: str, domain: str, analogy_style: str) -> str:
    return (
        f'Explain the concept of "{concept}" (domain: {domain}) as if you\'re talking to a '
        f"curious 7-year-old. Use a {analogy_style} analogy.\n\n"
        "Rules:\n"
        "- 3 to 6 sentences maximum\n"
        "- No words a 7-year-old wouldn't know, unless you immediately explain them\n"
        "- The analogy must be the heart of the explanation — not a decoration at the end\n"
        "- End with one sentence that hints at why this concept matters in real life\n\n"
        "Return your answer using the explain_concept tool."
    )


def generate_one(
    client: anthropic.Anthropic,
    concept: str,
    domain: str,
    analogy_style: str,
    model: str,
) -> tuple[dict, dict]:
    """Returns (record, usage) where usage has input_tokens and output_tokens."""
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[EXPLAIN_TOOL],
        tool_choice={"type": "tool", "name": "explain_concept"},
        messages=[
            {"role": "user", "content": build_user_prompt(concept, domain, analogy_style)}
        ],
    )

    tool_input = response.content[0].input
    record = {
        "concept": concept,
        "domain": domain,
        "analogy_style": analogy_style,
        "explanation": tool_input["explanation"],
        "analogy_type": tool_input["analogy_type"],
        "difficulty": tool_input["difficulty"],
        "key_analogy": tool_input["key_analogy"],
        "score": None,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return record, usage


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_PER_MTK.get(model, {"input": 1.0, "output": 5.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def load_checkpoint(output_path: Path) -> set[str]:
    """Return set of already-generated concept names."""
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
    parser.add_argument("--concepts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-concepts", type=int, default=2000)
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    args = parser.parse_args()

    concepts_path = Path(args.concepts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    concepts = []
    with concepts_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                concepts.append(json.loads(line))

    concepts = concepts[: args.max_concepts]
    done = load_checkpoint(output_path)
    print(f"Loaded {len(concepts)} concepts. {len(done)} already generated.", flush=True)

    client = anthropic.Anthropic()
    total_cost = 0.0
    total_tokens = 0
    generated = 0

    cost_log_path = output_path.parent / "cost_log.jsonl"

    with output_path.open("a") as out_f, cost_log_path.open("a") as cost_f:
        for i, row in enumerate(concepts):
            concept = row["concept"]
            if concept in done:
                continue

            domain = row.get("domain", "unknown")
            analogy_style = ANALOGY_STYLES[i % len(ANALOGY_STYLES)]

            try:
                record, usage = generate_one(client, concept, domain, analogy_style, args.model)
            except anthropic.APIError as e:
                print(f"API error on '{concept}': {e}", file=sys.stderr)
                continue

            cost = compute_cost(args.model, usage["input_tokens"], usage["output_tokens"])
            total_cost += cost
            total_tokens += usage["input_tokens"] + usage["output_tokens"]
            generated += 1

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            cost_log = {
                "concept": concept,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cost_usd": round(cost, 6),
                "cumulative_cost_usd": round(total_cost, 4),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            cost_f.write(json.dumps(cost_log) + "\n")
            cost_f.flush()

            if generated % 50 == 0:
                print(
                    f"[{generated}/{len(concepts) - len(done)}] "
                    f"${total_cost:.2f} spent | {total_tokens:,} tokens",
                    flush=True,
                )

            time.sleep(args.delay)

    print(f"\nDone. {generated} examples generated. Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
