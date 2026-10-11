"""Split curated QA by CFR part and near-duplicate answers, without training."""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_utils import check_split_overlap, load_examples
from project_config import load_config, runtime_paths


def part_key(example):
    url = example.get("metadata", {}).get("source_url", "")
    match = re.search(r"title-(\d+)/(?:section-(\d+)|part-(\d+))", url)
    if not match:
        raise ValueError(f"Missing recognizable source URL: {url}")
    title, section_part, explicit_part = match.groups()
    return f"title-{title}:part-{section_part or explicit_part}"


def answer_shingles(example):
    answer = example["messages"][-1]["content"].lower()
    # Normalize CFR citations, but preserve substantive numbers (fees, dates, etc.).
    answer = re.sub(r"§§?\s*[\d.]+(?:\([a-z0-9]+\))*", " section ", answer)
    answer = re.sub(r"\b[12]\s+cfr\s+(?:parts?\s+)?[\d.]+", " cfr reference ", answer)
    words = re.findall(r"[a-z0-9]+", answer)
    if len(words) < 12:
        return set()
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def create_split(examples, seed):
    keys = [part_key(example) for example in examples]
    parents = {key: key for key in keys}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    shingles = [answer_shingles(example) for example in examples]
    inverted = defaultdict(set)
    merges = []
    for index, current in enumerate(shingles):
        candidates = set()
        for shingle in current:
            candidates.update(inverted[shingle])
        for other in sorted(candidates):
            if find(keys[index]) == find(keys[other]):
                continue
            previous = shingles[other]
            similarity = len(current & previous) / len(current | previous)
            if similarity >= 0.85:
                merges.append({"parts": [keys[other], keys[index]],
                               "examples": [other, index], "jaccard": round(similarity, 4)})
                union(keys[index], keys[other])
        for shingle in current:
            inverted[shingle].add(index)

    grouped = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[find(key)].append(index)
    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(examples) * 0.1))

    def select(pool):
        # Subset-sum search balances counts while keeping each connected group intact.
        choices = {(0, 0): ()}
        for index, group in enumerate(pool):
            mask = 0
            for row in group:
                mask |= 1 << (int(keys[row].split(":")[0].removeprefix("title-")) - 1)
            for (count, existing_mask), selected in list(choices.items()):
                total = count + len(group)
                if total <= target * 2:
                    choices.setdefault((total, existing_mask | mask), selected + (index,))
        possible = [(state, selected) for state, selected in choices.items()
                    if state[0] > 0 and len(selected) < len(pool)]
        both_titles = [item for item in possible if item[0][1] == 3]
        options = both_titles or possible
        if not options:
            raise ValueError("Not enough independent groups for held-out evaluation")
        _, selected = min(options, key=lambda item: (abs(item[0][0] - target), len(item[1]), item[1]))
        selected = set(selected)
        return ([group for i, group in enumerate(pool) if i in selected],
                [group for i, group in enumerate(pool) if i not in selected])

    test_groups, remainder = select(groups)
    validation_groups, train_groups = select(remainder)
    partitions = {"train": sorted(i for group in train_groups for i in group),
                  "validation": sorted(i for group in validation_groups for i in group),
                  "test": sorted(i for group in test_groups for i in group)}
    memberships = [{keys[i] for i in partitions[name]} for name in partitions]
    assert all(not left & right for i, left in enumerate(memberships) for right in memberships[i + 1:])
    assert sorted(i for part in partitions.values() for i in part) == list(range(len(examples)))
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        check_split_overlap([examples[i] for i in partitions[left]], [examples[i] for i in partitions[right]])
    return partitions, keys, merges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    data = ROOT / "data"
    paths = runtime_paths(load_config(ROOT), ROOT)
    train_path = paths["train"]
    master_path = data / "all_examples.jsonl"
    input_path = master_path if master_path.exists() else train_path
    examples = load_examples(input_path)
    partitions, keys, merges = create_split(examples, args.seed)
    report = {
        "seed": args.seed, "total_examples": len(examples),
        "grouping": "CFR part; connected near-duplicate answer groups at 0.85 five-word shingle Jaccard",
        "limitations": "This detects lexical near-duplicates, not all semantic overlap or shared legal facts.",
        "near_duplicate_group_merges": merges,
        "splits": {name: {"count": len(indices),
                          "percent": round(len(indices) / len(examples) * 100, 2),
                          "source_parts": sorted({keys[i] for i in indices}),
                          "title_counts": dict(Counter(keys[i].split(":")[0] for i in indices)),
                          "original_example_indices": indices}
                   for name, indices in partitions.items()},
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    print(json.dumps({"total": len(examples), "counts": {n: len(p) for n, p in partitions.items()},
                      "near_duplicate_merges": len(merges),
                      "parts": {n: report["splits"][n]["source_parts"] for n in partitions}}, indent=2))
    if args.dry_run:
        return
    destinations = {name: paths[name] for name in ("train", "validation", "test")}
    for name in ("validation", "test"):
        if destinations[name].exists() and destinations[name].read_text().strip():
            raise FileExistsError(f"Refusing to overwrite nonempty {destinations[name].name}")
    if not master_path.exists():
        master_path.write_bytes(input_path.read_bytes())
    for name, indices in partitions.items():
        payload = "".join(json.dumps(examples[i], ensure_ascii=False) + "\n" for i in indices)
        destinations[name].write_text(payload, encoding="utf-8")
        report["splits"][name]["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    (data / "split_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Source-grouped split saved. No training performed.")


if __name__ == "__main__":
    main()
