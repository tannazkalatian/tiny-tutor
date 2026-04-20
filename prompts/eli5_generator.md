# ELI5 Generator Prompt

Version: v1.0
Last updated: 2025-01-01

---

## System Prompt

```
You are an expert science communicator who specializes in explaining complex STEM concepts
to children aged 5–10. Your explanations are warm, vivid, and always grounded in a concrete
analogy drawn from the child's everyday world. You never use jargon without immediately
replacing it with something simpler.
```

---

## User Prompt Template

```
Explain the concept of "{{concept}}" (domain: {{domain}}) as if you're talking to a
curious 7-year-old. Use a {{analogy_style}} analogy.

Rules:
- 3 to 6 sentences maximum
- No words a 7-year-old wouldn't know, unless you immediately explain them in the same sentence
- The analogy must be the heart of the explanation — not a decoration at the end
- End with one sentence that hints at why this concept matters in real life

Return your answer using the explain_concept tool.
```

---

## Tool Schema (passed to Claude via tool_use)

```json
{
  "name": "explain_concept",
  "description": "Return a structured ELI5 explanation of a STEM concept.",
  "input_schema": {
    "type": "object",
    "properties": {
      "explanation": {
        "type": "string",
        "description": "The ELI5 explanation, 3–6 sentences."
      },
      "analogy_type": {
        "type": "string",
        "enum": ["animals", "food", "everyday_objects", "sports", "building"],
        "description": "The analogy category actually used."
      },
      "difficulty": {
        "type": "string",
        "enum": ["easy", "medium", "hard"],
        "description": "How hard this concept is for a 7-year-old to grasp."
      },
      "key_analogy": {
        "type": "string",
        "description": "The specific analogy object used (e.g. 'pizza', 'anthill', 'Lego brick')."
      }
    },
    "required": ["explanation", "analogy_type", "difficulty", "key_analogy"]
  }
}
```

---

## Self-Critique Prompt Template

Used in `src/filter.py` to score each generated explanation.

```
Rate the following ELI5 explanation of "{{concept}}" on three criteria.
Be strict — a 4 means genuinely good, not just okay.

Explanation:
"""
{{explanation}}
"""

Use the rate_explanation tool.
```

### Self-Critique Tool Schema

```json
{
  "name": "rate_explanation",
  "description": "Rate an ELI5 explanation on three criteria.",
  "input_schema": {
    "type": "object",
    "properties": {
      "clarity": {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": "Would a 7-year-old actually follow this? 5 = yes, completely. 1 = lost after sentence 1."
      },
      "accuracy": {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": "Is it factually correct? Simplification is fine; distortion is not. 5 = correct. 1 = misleading."
      },
      "age_appropriateness": {
        "type": "integer",
        "minimum": 1,
        "maximum": 5,
        "description": "No unexplained jargon, no condescension, right tone. 5 = perfect. 1 = wrong register entirely."
      },
      "reasoning": {
        "type": "string",
        "description": "One sentence explaining the lowest score given."
      }
    },
    "required": ["clarity", "accuracy", "age_appropriateness", "reasoning"]
  }
}
```

### Score Computation

```
composite = 0.4 * clarity + 0.4 * accuracy + 0.2 * age_appropriateness
```

Keep if `composite >= 4.0`.

---

## Analogy Style Rotation

To ensure dataset diversity without tripling API costs, analogy styles are rotated
across concepts rather than generating multiple styles per concept:

```python
ANALOGY_STYLES = ["animals", "food", "everyday_objects", "sports", "building"]
style = ANALOGY_STYLES[concept_index % len(ANALOGY_STYLES)]
```

---

## Versioning Notes

- **v1.0** — initial prompt, tool use for structured output, 5-style rotation
- Iterate on this file; the generation script reads the prompt from here at runtime
  so prompt changes are automatically reflected without touching `src/generate.py`
