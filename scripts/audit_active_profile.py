"""Audit every active profile row against local sources; optionally repair templates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import build_detailed_ecfr as builder
from project_config import load_config, runtime_paths
from training_utils import load_examples, check_split_overlap


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template_issues(question, metadata):
    issues = []
    actor = re.search(r"what (?:is (.+?) permitted to do|requirement applies to (.+?) concerning|restriction applies to (.+?) concerning)", question)
    if actor:
        subject = next(value for value in actor.groups() if value is not None)
        if re.search(r"\b(that|which|who|whenever|when|if)\b|[—:]|\.\s+[A-Z]", subject, re.I):
            issues.append("clause_or_caption_as_actor")
        if re.match(r"(?:such|these|this|those|it\b|the provisions of this|any other information)", subject, re.I):
            issues.append("unresolved_actor_context")
        if metadata.get("question_type") == "permission" and re.match(r"(?:costs|information|personal appearance|testimony|records|documents|applications|payments|fees|requests|reports)\b", subject, re.I):
            issues.append("inanimate_permission_subject")
        if metadata.get("question_type") == "permission" and re.search(r"concerning 'be\b", question):
            issues.append("passive_action_as_permission")
    if not question.rstrip().endswith("?"):
        issues.append("question_not_interrogative")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    config = load_config(ROOT)
    paths = runtime_paths(config, ROOT)
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], local_files_only=True)
    source = json.loads((ROOT / "data/detailed_ecfr/source_manifest.json").read_text())
    units = {unit["id"]: unit for unit in builder.collect_units(source)}
    limit = config["training"]["max_seq_length"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {"method": "All-row source, token, duplicate and template audit; semantic heuristics are not independent legal certification.", "apply": args.apply, "splits": {}, "findings": []}
    prepared = {}

    def tokens(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False, **config["model"]["chat_template_kwargs"])
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    for split in ("train", "validation"):
        rows = load_examples(paths[split])
        counts = Counter()
        kept = []
        seen = set()
        for index, row in enumerate(rows):
            meta = row["metadata"]
            question, answer = [message["content"] for message in row["messages"]]
            unit = units.get(meta.get("source_id"))
            if unit is None:
                raise ValueError(f"Missing source unit: {split}:{index}")
            evidence = meta.get("source_paragraph_indices", [])
            if not evidence and meta.get("curation_method") == "reviewed":
                evidence = [node["index"] for node in unit["nodes"]]
                meta.update(source_paragraph_indices=evidence, supporting_quotes=[unit["nodes"][i]["text"] for i in evidence], source_url=unit["url"], snapshot_date=source["date"])
                meta["support_sha256"] = hashlib.sha256("\n".join(meta["supporting_quotes"]).encode()).hexdigest()
                counts["reviewed_evidence_backfilled"] += 1
            quotes = [unit["nodes"][i]["text"] for i in evidence]
            if not quotes or quotes != meta.get("supporting_quotes") or hashlib.sha256("\n".join(quotes).encode()).hexdigest() != meta.get("support_sha256"):
                raise ValueError(f"Invalid source evidence: {split}:{index}")
            issues = template_issues(question, meta)
            if meta.get("curation_method") != "reviewed":
                # A removed colon can hide a dangling list lead-in even if the final answer ends normally.
                answer_lines = {builder.normalized(line) for line in answer.splitlines()}
                if any(quote.rstrip().endswith(":") and builder.normalized(builder.clean(quote).rstrip(":")) in answer_lines for quote in quotes):
                    issues.append("stripped_list_lead_in")
                source_text = builder.normalized(" ".join(builder.clean(quote) for quote in quotes)).lower()
                qualifications = re.findall(r"\b(?:unless|except|provided that|notwithstanding|however|only if|subject to)\b", source_text)
                if meta.get("question_type") != "table_lookup" and any(phrase not in answer.lower() for phrase in qualifications):
                    issues.append("source_qualification_omitted")
                sentences = [builder.normalized(line).lower() for line in answer.splitlines() if line.strip()]
                if any(sentence not in source_text and sentence.rstrip("—–:") not in source_text for sentence in sentences):
                    issues.append("answer_not_contiguous_source_span")
            if issues:
                counts.update(issues)
                finding = {"split": split, "index": index, "citation": meta.get("citation"), "issues": issues, "question_before": question, "answer_before": answer}
                if meta.get("curation_method") != "reviewed":
                    question_issues = [issue for issue in issues if issue not in ("stripped_list_lead_in", "answer_not_contiguous_source_span", "source_qualification_omitted")]
                    if question_issues:
                        anchor_match = re.search(r"concerning '(.+)'", question)
                        anchor = anchor_match.group(1) if anchor_match else " ".join(answer.splitlines()[-1].split()[:12]).rstrip(".,;:")
                        row["messages"][0]["content"] = f"Under {meta['citation']}, what does the rule state concerning '{anchor}'?"
                        focused = {
                            "2 CFR § 2701.503": "Under 2 CFR § 2701.503, may financial-statement audit costs be included in indirect cost rate negotiations, and under which provision?",
                            "5 CFR § 1655.12": "Under 5 CFR § 1655.12(d)(2), what additional information must accompany the signed loan agreement?",
                            "8 CFR § 341.2": "Under 8 CFR § 341.2(a)(1), under what documentation conditions may a citizenship application be processed without an interview?",
                        }
                        row["messages"][0]["content"] = focused.get(meta["citation"], row["messages"][0]["content"])
                    row["messages"][1]["content"] = "\n".join(builder.clean(quote) for quote in quotes)
                    if question_issues:
                        meta["question_type"] = "source_scoped_rule"
                    meta["semantic_review"] = "template_repaired_against_source_not_independent_legal_certification"
                    # Never teach clipped context to meet a token budget.
                    if tokens(row) > limit:
                        finding["action"] = "excluded_overlength_complete_context"
                        counts["excluded_overlength_complete_context"] += 1
                        report["findings"].append(finding)
                        continue
                    finding.update(action="repaired_question_and_complete_evidence", question_after=row["messages"][0]["content"], answer_after=row["messages"][1]["content"])
                    counts["repaired"] += 1
                else:
                    finding["action"] = "reviewed_row_flagged_not_automatically_changed"
                report["findings"].append(finding)
            meta["tokens"] = tokens(row)
            if meta["tokens"] > limit:
                raise ValueError(f"Overlength row: {split}:{index}")
            key = builder.normalized(row["messages"][0]["content"]).casefold()
            if key in seen:
                label = ", ".join(unit["nodes"][i].get("label", "") or str(i) for i in evidence)
                row["messages"][0]["content"] = row["messages"][0]["content"].rstrip("?") + f" (specifically source paragraphs {label})?"
                key = builder.normalized(row["messages"][0]["content"]).casefold()
                if key in seen:
                    previous = next(item for item in kept if builder.normalized(item["messages"][0]["content"]).casefold() == key)
                    if previous["messages"][1]["content"] == row["messages"][1]["content"]:
                        counts["redundant_repaired_pair_removed"] += 1
                        continue
                    raise ValueError(f"Conflicting normalized duplicate: {split}:{index}")
                meta["tokens"] = tokens(row)
                if meta["tokens"] > limit:
                    counts["excluded_overlength_disambiguation"] += 1
                    continue
                counts["question_disambiguated"] += 1
            seen.add(key)
            kept.append(row)
        prepared[split] = kept
        report["splits"][split] = {"before": len(rows), "after": len(kept), "counts": dict(counts), "max_tokens": max(r["metadata"]["tokens"] for r in kept)}
    test_hash = digest(paths["test"])
    test = load_examples(paths["test"])
    check_split_overlap(prepared["train"], prepared["validation"])
    for split in prepared:
        check_split_overlap(prepared[split], test)
    report["test_sha256"] = test_hash
    if args.apply:
        backup = ROOT / "data/history" / (stamp + "-profile-audit")
        backup.mkdir(parents=True)
        manifest_path = paths["train"].parent / "manifest.json"
        for path in [paths["train"], paths["validation"], manifest_path, ROOT / "config.local.json"]:
            shutil.copy2(path, backup / path.name)
        for split, rows in prepared.items():
            staged = paths[split].with_suffix(".jsonl.pending")
            staged.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
            load_examples(staged)
            staged.replace(paths[split])
        manifest = json.loads(manifest_path.read_text())
        manifest["active_file_hashes"] = {split: digest(paths[split]) for split in prepared}
        for split, rows in prepared.items():
            manifest["counts"][split] = len(rows)
            manifest[split]["selected_by_title"] = dict(sorted(Counter(str(int(row["metadata"]["source_id"].split(":")[0].split("-")[1])) for row in rows).items(), key=lambda pair: int(pair[0])))
            manifest[split]["selected_parts"] = len({row["metadata"]["source_part"] for row in rows})
            manifest[split]["reviewed_selected"] = sum(row["metadata"].get("curation_method") == "reviewed" for row in rows)
        manifest["mean_train_tokens"] = sum(row["metadata"]["tokens"] for row in prepared["train"]) / len(prepared["train"])
        manifest["latest_all_row_audit"] = {"timestamp": stamp, "report": "audit_report.json", "method": report["method"]}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        local = ROOT / "config.local.json"
        local_config = json.loads(local.read_text())
        training = local_config["training"]
        training["max_steps"] = math.ceil(len(prepared["train"]) / (training.get("per_device_train_batch_size", 1) * training["gradient_accumulation_steps"]))
        local.write_text(json.dumps(local_config, indent=2) + "\n")
        assert digest(paths["test"]) == test_hash
    report_path = paths["train"].parent / ("audit_report.json" if args.apply else "audit_preview.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["splits"], indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
