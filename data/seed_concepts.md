# Seed Concepts

## Source

STEM concepts are sourced from the following Wikipedia glossary pages.
These pages list terms with short definitions, making them easy to scrape
and clean into a flat concept list.

### Physics
- https://en.wikipedia.org/wiki/Glossary_of_physics
- https://en.wikipedia.org/wiki/Glossary_of_classical_mechanics
- https://en.wikipedia.org/wiki/Glossary_of_quantum_mechanics
- https://en.wikipedia.org/wiki/Glossary_of_thermodynamics

### Biology
- https://en.wikipedia.org/wiki/Glossary_of_biology
- https://en.wikipedia.org/wiki/Glossary_of_genetics
- https://en.wikipedia.org/wiki/Glossary_of_ecology
- https://en.wikipedia.org/wiki/Glossary_of_cell_biology

### Computer Science
- https://en.wikipedia.org/wiki/Glossary_of_computer_science
- https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence
- https://en.wikipedia.org/wiki/Glossary_of_graph_theory

### Math
- https://en.wikipedia.org/wiki/Glossary_of_mathematics
- https://en.wikipedia.org/wiki/Glossary_of_calculus
- https://en.wikipedia.org/wiki/Glossary_of_probability_and_statistics

---

## Cleaning Notes

After scraping, concepts are filtered by:
1. **Length** — drop anything under 3 characters or over 60 characters
2. **Proper nouns** — drop named theorems and named people (e.g. "Newton's law", "Euler")
   unless the concept itself is general (e.g. "gravity" is kept, "Newton" is dropped)
3. **Duplicates** — deduplicate across domains; prefer the biology/physics/cs/math label
   for the domain that most naturally "owns" the concept
4. **Difficulty** — manually tag ~50 concepts as "hard" to ensure the dataset covers
   advanced topics (e.g. "entropy", "eigenvector", "meiosis") not just easy ones

Final list is stored in `data/raw/seed_concepts.jsonl` (one concept per line):

```json
{"concept": "photosynthesis", "domain": "biology", "difficulty": "easy"}
{"concept": "eigenvalue", "domain": "math", "difficulty": "hard"}
```

---

## Target Distribution

| Domain | Count |
|---|---|
| Biology | 500 |
| Physics | 500 |
| Computer Science | 500 |
| Math | 500 |
| **Total** | **2,000** |

Difficulty split target: ~60% easy, ~30% medium, ~10% hard.
