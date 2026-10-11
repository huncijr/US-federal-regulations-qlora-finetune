"""Add natural, applied-case, and clarification variants within each frozen split."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import load_config, runtime_paths

CITED_PREFIX = re.compile(r"^under\s+.+?,\s+(?=[A-Z0-9a-z])", re.IGNORECASE)

CLARIFICATIONS = [
    ("What does Agency mean?",
     "Which context do you mean? If you are asking about 1 CFR § 1.1, I can give that regulatory definition with its exclusions. Otherwise, please specify the provision or situation you have in mind."),
    ("What does Document mean?",
     "Which context do you mean? If you are asking about 1 CFR § 1.1, I can give that regulatory definition. Otherwise, please specify the provision or situation you have in mind."),
    ("What does Filing mean?",
     "Which filing process do you mean? If you are asking about 1 CFR § 1.1, I can give that definition. Otherwise, please name the agency, court, or regulation involved."),
    ("What are the office hours?",
     "Which office do you mean? Office hours differ by office. If you mean the Office of the Federal Register under 1 CFR § 2.3(d), I can give those hours. Otherwise, please specify the office."),
    ("Where is the office located?",
     "Which office do you mean? If you mean the Office of the Federal Register under 1 CFR § 2.3(b), I can give that location. Otherwise, please specify the office."),
    ("Who is the secretary?",
     "Secretary of which body do you mean? If you mean the Administrative Committee of the Federal Register under 1 CFR § 2.2, I can answer that. Otherwise, please specify the committee or organization."),
    ("Who is the chairman?",
     "Chairman of which body do you mean? If you mean the Administrative Committee of the Federal Register under 1 CFR § 2.2, I can answer that. Otherwise, please specify the committee or organization."),
    ("Is there any difference between a regulation and a rule?",
     "In which regulation or chapter do you mean? For example, under 1 CFR § 1.1 they have the same meaning unless context requires otherwise. Please specify the provision if you mean a different context."),
    ("Can I reproduce this Federal Register material?",
     "Which material and edition do you mean? For example, under 1 CFR § 2.6 any person may reproduce or republish material in regular or special Federal Register editions without restriction. Please specify the material if you mean something else."),
    ("When is my document considered filed?",
     "Filed with which office and under which regulation do you mean? For example, under 1 CFR § 1.1 a document is filed only after receipt, processing, and assignment of a publication date under Part 17. Please specify your context."),
    ("What does this provision require?",
     "Which provision do you mean? Please give the CFR citation or describe the situation, so I can answer from the correct section instead of guessing."),
    ("What should I do in this case?",
     "Could you describe the situation and name the applicable regulation, if known? I should not guess which federal rule applies, so please provide the provision or the key facts."),
]


def naturalize(question):
    match = CITED_PREFIX.match(question.strip())
    if not match:
        return None
    rest = question.strip()[match.end():]
    if not rest:
        return None
    rest = rest[0].upper() + rest[1:]
    if not rest.endswith("?"):
        rest += "?"
    return rest


def case_frame(natural, citation):
    return (
        f"A coworker asks you at work: \"{natural}\" "
        f"Give the correct federal-regulatory answer."
    )


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_variants(examples):
    out = []
    stats = {"natural": 0, "case": 0, "skipped": 0}
    for example in examples:
        question = example["messages"][0]["content"]
        answer = example["messages"][-1]["content"]
        natural = naturalize(question)
        if natural is None or natural == question:
            stats["skipped"] += 1
            continue
        base_meta = dict(example.get("metadata", {}))
        citation = base_meta.get("citation", "the applicable provision")
        natural_meta = dict(base_meta)
        natural_meta.update({"variant": "natural", "derived_from": base_meta.get("source_id", citation),
                             "question_method": "rule_based_citation_strip"})
        out.append({"messages": [{"role": "user", "content": natural},
                                  {"role": "assistant", "content": answer}],
                    "metadata": natural_meta})
        stats["natural"] += 1
        framed = case_frame(natural, citation)
        case_meta = dict(base_meta)
        case_meta.update({"variant": "applied_case", "derived_from": base_meta.get("source_id", citation),
                          "question_method": "rule_based_scenario_frame"})
        out.append({"messages": [{"role": "user", "content": framed},
                                  {"role": "assistant", "content": answer}],
                    "metadata": case_meta})
        stats["case"] += 1
    return out, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    data = ROOT / "data"
    config = load_config(ROOT)
    paths = runtime_paths(config, ROOT)
    files = {name: paths[name] for name in ("train", "validation", "test")}
    splits = {name: load(path) for name, path in files.items()}
    variants = {name: build_variants(rows) for name, rows in splits.items()}
    total_new = sum(len(v[0]) for v in variants.values())
    print(json.dumps({name: {"base": len(splits[name]), "new": len(variants[name][0]),
                             "stats": variants[name][1]} for name in splits}, indent=2))
    clarifications = [{"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}],
                       "metadata": {"variant": "clarification", "citation": "multiple/unspecified",
                                    "answer_verification": "behavioral_clarification",
                                    "question_review": "hand_written"}} for q, a in CLARIFICATIONS]
    print(f"Clarification examples (train only): {len(clarifications)}")
    # Collision check: new questions must not already exist anywhere.
    # Identical natural questions from different sections are disambiguated
    # with a citation parenthetical rather than dropped.
    existing_questions = {m["content"] for rows in splits.values() for e in rows for m in e["messages"] if m["role"] == "user"}
    for name, (new_rows, _) in variants.items():
        for row in new_rows:
            question = row["messages"][0]["content"]
            if question in existing_questions:
                citation = row["metadata"].get("citation", "")
                fixed = question[:-1] + f" ({citation})?" if question.endswith("?") else f"{question} ({citation})"
                if fixed in existing_questions:
                    raise RuntimeError(f"Variant collides even after disambiguation in {name}: {question}")
                row["messages"][0]["content"] = fixed
                row["metadata"]["variant_disambiguated"] = True
                question = fixed
            existing_questions.add(question)
    for row in clarifications:
        question = row["messages"][0]["content"]
        if question in existing_questions:
            raise RuntimeError(f"Clarification collides: {question}")
        existing_questions.add(question)
    if not args.apply:
        print("Dry run only; no files changed.")
        return
    for name, (new_rows, _) in variants.items():
        with files[name].open("a", encoding="utf-8") as output:
            for row in new_rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with files["train"].open("a", encoding="utf-8") as output:
        for row in clarifications:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Token-length guard on everything added.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], local_files_only=True)
    failures = []
    for name, path in files.items():
        for row in load(path):
            messages = row["messages"]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                                      add_generation_prompt=False, **config["model"]["chat_template_kwargs"])
            tokens = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
            if tokens > 512:
                failures.append((name, messages[0]["content"][:80], tokens))
    if failures:
        raise RuntimeError(f"Overlength examples: {failures}")
    manifest_path = data / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["augmentation"] = {
        "method": "rule-based natural/case variants kept in their source split; hand-written clarifications in train",
        "added": {name: len(variants[name][0]) for name in splits},
        "clarifications_in_train": len(clarifications),
        "token_budget": "all messages <= 512 tokens",
    }
    for name, path in files.items():
        manifest["splits"][name]["count"] = len(load(path))
        manifest["splits"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Applied. Total new: {total_new + len(clarifications)}.")


if __name__ == "__main__":
    main()
