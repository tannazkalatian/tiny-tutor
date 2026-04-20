"""
Eval harness: LLM-as-judge + readability scores + head-to-head comparison.

Usage:
    python src/evaluate.py \
        --base-model HuggingFaceTB/SmolLM2-360M-Instruct \
        --finetuned-model checkpoints/final \
        --test-set data/test_concepts.jsonl \
        --output results/eval_report.json
"""

import argparse
import json
import random
import time
from pathlib import Path

import anthropic
import textstat

JUDGE_TOOL = {
    "name": "judge_pair",
    "description": "Compare two ELI5 explanations of the same concept.",
    "input_schema": {
        "type": "object",
        "properties": {
            "winner": {
                "type": "string",
                "enum": ["A", "B", "tie"],
                "description": "Which explanation is better overall for a 7-year-old?",
            },
            "clarity_a": {"type": "integer", "minimum": 1, "maximum": 5},
            "clarity_b": {"type": "integer", "minimum": 1, "maximum": 5},
            "accuracy_a": {"type": "integer", "minimum": 1, "maximum": 5},
            "accuracy_b": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the winner choice.",
            },
        },
        "required": ["winner", "clarity_a", "clarity_b", "accuracy_a", "accuracy_b", "reasoning"],
    },
}

JUDGE_SYSTEM = (
    "You are a strict evaluator of children's educational content. "
    "You will see two explanations of the same concept, labeled A and B. "
    "Judge which is better for a curious 7-year-old. The labels are randomized — "
    "ignore any stylistic differences unrelated to quality."
)


def generate_explanation(model, tokenizer, concept: str) -> str:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly science tutor. Explain STEM concepts simply, "
            "using analogies a 7-year-old would understand.",
        },
        {"role": "user", "content": f"Explain '{concept}' like I'm 5."},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.7, do_sample=True)
    decoded = tokenizer.decode(outputs[0][input_ids.shape[1] :], skip_special_tokens=True)
    return decoded.strip()


def judge_pair(client: anthropic.Anthropic, concept: str, exp_a: str, exp_b: str, model: str) -> dict:
    prompt = (
        f'Concept: "{concept}"\n\n'
        f"Explanation A:\n{exp_a}\n\n"
        f"Explanation B:\n{exp_b}\n\n"
        "Which explanation would a 7-year-old find clearer and more accurate?"
    )
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "judge_pair"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].input


def readability_stats(text: str) -> dict:
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 1),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 1),
        "gunning_fog": round(textstat.gunning_fog(text), 1),
        "word_count": len(text.split()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--finetuned-model", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--output", default="results/eval_report.json")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Install with: pip install unsloth") from e

    test_concepts = []
    with Path(args.test_set).open() as f:
        for line in f:
            line = line.strip()
            if line:
                test_concepts.append(json.loads(line))
    test_concepts = test_concepts[: args.max_examples]
    print(f"Evaluating on {len(test_concepts)} test concepts.")

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        args.base_model, max_seq_length=512, load_in_4bit=True
    )

    print(f"Loading fine-tuned model: {args.finetuned_model}")
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        args.finetuned_model, max_seq_length=512, load_in_4bit=True
    )

    client = anthropic.Anthropic()
    results = []
    wins = {"base": 0, "finetuned": 0, "tie": 0}

    for i, row in enumerate(test_concepts):
        concept = row["concept"]
        base_exp = generate_explanation(base_model, base_tok, concept)
        ft_exp = generate_explanation(ft_model, ft_tok, concept)

        # Randomize which is A/B to avoid position bias
        flip = random.random() < 0.5
        exp_a = ft_exp if flip else base_exp
        exp_b = base_exp if flip else ft_exp
        label_a = "finetuned" if flip else "base"
        label_b = "base" if flip else "finetuned"

        try:
            judgment = judge_pair(client, concept, exp_a, exp_b, args.judge_model)
        except anthropic.APIError as e:
            print(f"Judge API error on '{concept}': {e}")
            continue

        raw_winner = judgment["winner"]
        if raw_winner == "tie":
            winner = "tie"
        elif raw_winner == "A":
            winner = label_a
        else:
            winner = label_b

        wins[winner] += 1

        result = {
            "concept": concept,
            "base_explanation": base_exp,
            "finetuned_explanation": ft_exp,
            "winner": winner,
            "judgment": judgment,
            "readability_base": readability_stats(base_exp),
            "readability_finetuned": readability_stats(ft_exp),
        }
        results.append(result)

        if (i + 1) % 20 == 0:
            total = sum(wins.values())
            print(
                f"[{i+1}/{len(test_concepts)}] "
                f"finetuned={wins['finetuned']} base={wins['base']} tie={wins['tie']} "
                f"finetuned_win_rate={wins['finetuned']/total*100:.1f}%"
            )
        time.sleep(args.delay)

    total = sum(wins.values())
    summary = {
        "n": total,
        "finetuned_wins": wins["finetuned"],
        "base_wins": wins["base"],
        "ties": wins["tie"],
        "finetuned_win_rate": round(wins["finetuned"] / total, 3) if total else 0,
        "avg_readability_finetuned": round(
            sum(r["readability_finetuned"]["flesch_reading_ease"] for r in results) / len(results), 1
        ) if results else 0,
        "avg_readability_base": round(
            sum(r["readability_base"]["flesch_reading_ease"] for r in results) / len(results), 1
        ) if results else 0,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nEval complete. Report saved to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
