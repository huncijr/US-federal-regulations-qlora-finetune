"""Create a compact, title/part-balanced profile from existing frozen eCFR splits."""

import argparse
import hashlib
import heapq
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import load_config, runtime_paths
from training_utils import check_split_overlap, load_examples

TYPES = {"definition", "contents", "obligation", "restriction", "condition_exception",
         "permission", "timing", "value_requirement", "section_question", "cross_reference", "table_lookup"}


def eligible(row):
    meta = row["metadata"]
    if meta.get("curation_method") == "reviewed":
        return meta["tokens"] <= 512
    if meta["question_type"] not in TYPES or meta["tokens"] > 320:
        return False
    question, answer = [message["content"] for message in row["messages"]]
    if len(answer) >= 25 and answer.lower() in question.lower():
        return False
    return True


def title_of(row):
    return int(row["metadata"]["source_id"].split(":")[0].removeprefix("title-"))


def allocate(counts, limit):
    quotas = {title: 0 for title in counts}
    left = min(limit, sum(counts.values()))
    while left:
        changed = False
        for title in sorted(counts):
            if left and quotas[title] < counts[title]:
                quotas[title] += 1
                left -= 1
                changed = True
        if not changed:
            break
    return quotas


def select(path, limit, seed):
    counts, part_counts, reviewed = Counter(), Counter(), []
    for line in path.open():
        row = json.loads(line)
        if not eligible(row):
            continue
        title = title_of(row)
        part = row["metadata"]["source_part"]
        counts[title] += 1
        part_counts[part] += 1
        if row["metadata"].get("curation_method") == "reviewed":
            reviewed.append(row)
    quotas = allocate(counts, limit)
    parts_by_title = defaultdict(list)
    for part in sorted(part_counts):
        title = int(part.split(":")[0].removeprefix("title-"))
        parts_by_title[title].append(part)
    capacities = {part: max(16, 2 * math.ceil(quotas[title] / len(parts)) + 16)
                  for title, parts in parts_by_title.items() for part in parts}
    heaps = defaultdict(list)
    title_heaps = defaultdict(list)
    for index, line in enumerate(path.open()):
        row = json.loads(line)
        if not eligible(row):
            continue
        part = row["metadata"]["source_part"]
        score = int(hashlib.sha256((str(seed) + row["messages"][0]["content"]).encode()).hexdigest()[:16], 16)
        heap = heaps[part]
        item = (-score, index, row)
        title = title_of(row)
        title_heap = title_heaps[title]
        title_capacity = max(1, 2 * quotas[title])
        if len(title_heap) < title_capacity:
            heapq.heappush(title_heap, item)
        elif score < -title_heap[0][0]:
            heapq.heapreplace(title_heap, item)
        if len(heap) < capacities[part]:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    queues = {part: deque(item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1])))
              for part, heap in heaps.items()}
    result, seen, selected_counts = [], set(), Counter()
    for row in reviewed:
        title = title_of(row)
        question = row["messages"][0]["content"]
        if question not in seen and selected_counts[title] < quotas[title]:
            seen.add(question)
            result.append(row)
            selected_counts[title] += 1
    while any(selected_counts[title] < quotas[title] for title in quotas):
        changed = False
        for title, parts in sorted(parts_by_title.items()):
            for part in parts:
                if selected_counts[title] >= quotas[title]:
                    break
                queue = queues[part]
                while queue and queue[0]["messages"][0]["content"] in seen:
                    queue.popleft()
                if queue:
                    row = queue.popleft()
                    seen.add(row["messages"][0]["content"])
                    result.append(row)
                    selected_counts[title] += 1
                    changed = True
        if not changed:
            break
    # Sparse parts may not fill their share; use the title-wide deterministic
    # reserve after the part round-robin has achieved broad representation.
    for title, heap in sorted(title_heaps.items()):
        for _, _, row in sorted(heap, key=lambda item: (-item[0], item[1])):
            if selected_counts[title] >= quotas[title]:
                break
            question = row["messages"][0]["content"]
            if question not in seen:
                seen.add(question)
                result.append(row)
                selected_counts[title] += 1
    if len(result) != min(limit, sum(counts.values())):
        raise RuntimeError("Sampling capacity insufficient; no active config was changed.")
    return result, {"eligible_by_title": dict(counts), "selected_by_title": dict(selected_counts),
                    "eligible_parts": len(part_counts),
                    "selected_parts": len({row["metadata"]["source_part"] for row in result}),
                    "reviewed_selected": sum(row["metadata"].get("curation_method") == "reviewed" for row in result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--train-size", type=int, default=8400)
    parser.add_argument("--validation-size", type=int, default=400)
    args = parser.parse_args()
    cfg = load_config(ROOT)
    paths = runtime_paths(cfg, ROOT)
    # The full files stay unchanged; subsequent invocations reuse their recorded paths.
    target = ROOT / "data" / "one_hour"
    previous_manifest = target / "manifest.json"
    full = {name: paths[name] for name in ("train", "validation", "test")}
    if previous_manifest.exists():
        previous = json.loads(previous_manifest.read_text())
        full = {name: ROOT / value for name, value in previous["full_source_files"].items()}
    train, train_stats = select(full["train"], args.train_size, cfg["training"]["seed"])
    validation, validation_stats = select(full["validation"], args.validation_size, cfg["training"]["seed"])
    check_split_overlap(train, validation)
    train_parts = {row["metadata"]["source_part"] for row in train}
    validation_parts = {row["metadata"]["source_part"] for row in validation}
    assert not train_parts & validation_parts
    target.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (target / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        load_examples(target / f"{name}.jsonl")
    manifest = {"profile": "approximately_one_hour", "seed": cfg["training"]["seed"],
                "full_source_files": {name: str(path.relative_to(ROOT)) for name, path in full.items()},
                "full_source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in full.items()},
                "train": train_stats, "validation": validation_stats,
                "counts": {"train": len(train), "validation": len(validation)},
                "mean_train_tokens": sum(row["metadata"]["tokens"] for row in train) / len(train),
                "training_started": False,
                "limitations": "Balanced coverage does not mean all source facts were trained or factual accuracy is guaranteed. Validation titles without available frozen-source examples are not invented."}
    previous_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.apply:
        local_file = ROOT / "config.local.json"
        backup = target / "config.before-profile.json"
        if not backup.exists():
            backup.write_bytes(local_file.read_bytes())
        local = json.loads(local_file.read_text())
        local["data"] = {"train": str((target / "train.jsonl").relative_to(ROOT)),
                         "validation": str((target / "validation.jsonl").relative_to(ROOT)),
                         "test": str(full["test"].relative_to(ROOT))}
        local["training"].update(max_steps=math.ceil(len(train) / local["training"]["gradient_accumulation_steps"]),
                                  eval_steps=350, eval_max_samples=400,
                                  max_wall_seconds=3600, save_reserve_seconds=300,
                                  final_eval_full=False)
        local_file.write_text(json.dumps(local, indent=2), encoding="utf-8")
        print("Activated bounded-time profile. Full datasets and the frozen final test remain unchanged.")


if __name__ == "__main__":
    main()
