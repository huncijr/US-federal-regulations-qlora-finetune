"""Snapshot complete eCFR titles, verify references, and test the configured model."""

import argparse
import hashlib
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import load_config, runtime_paths

CONFIG = load_config(ROOT)
PATHS = runtime_paths(CONFIG, ROOT)
DATA = ROOT / "data" / "ecfr_baseline"


def text(element):
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_units(root, title, date):
    records, inventory = [], []

    def visit(element, path):
        kind = element.get("TYPE")
        if kind:
            path = path + [{"type": kind, "number": element.get("N"),
                            "heading": text(element.find("HEAD"))
                            if element.find("HEAD") is not None else ""}]
        if kind in ("SECTION", "APPENDIX"):
            heading = path[-1]["heading"]
            blocks = [text(child) for child in element if child.tag != "HEAD"]
            reference = "\n\n".join(block for block in blocks if block)
            reserved = "[reserved]" in heading.lower() and not reference
            part = next((p["number"] for p in reversed(path) if p["type"] == "PART"), None)
            number = element.get("N", "")
            base = f"https://www.ecfr.gov/on/{date}/title-{title}"
            url = f"{base}/section-{number}" if kind == "SECTION" else f"{base}/part-{part}"
            identifier = f"title-{title}:{kind.lower()}:{number}"
            inventory.append({"id": identifier, "path": path, "url": url,
                              "reserved": reserved, "has_content": bool(reference)})
            if reference:
                citation = f"{title} CFR § {number}" if kind == "SECTION" else f"{title} CFR {heading}"
                topic = re.sub(r"^§\s*\S+\s*", "", heading).strip()
                question = (
                    f"According to {citation}, {topic[0].lower() + topic[1:]}"
                    if topic.endswith("?") and topic else
                    f"Under {citation} ({topic}), what does this provision state? "
                    "Give the applicable requirements, definitions, or procedures, "
                    "including relevant qualifications."
                )
                records.append({"id": identifier, "title": title, "kind": kind,
                                "number": number, "heading": heading, "path": path,
                                "source_url": url, "snapshot_date": date,
                                "question": question, "reference_answer": reference,
                                "reference_sha256": hashlib.sha256(reference.encode()).hexdigest(),
                                "reference_type": "verbatim_xml_text_whitespace_normalized",
                                "context_supplied_to_model": False})
            return
        for child in element:
            visit(child, path)

    visit(root, [])
    return records, inventory


def prepare():
    import requests

    DATA.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Federal-regulations-QLoRA research/1.0"
    response = session.get("https://www.ecfr.gov/api/versioner/v1/titles.json", timeout=90)
    response.raise_for_status()
    titles = [t for t in response.json()["titles"] if t["number"] in (1, 2)]
    date = min(t["up_to_date_as_of"] for t in titles)
    all_records, all_inventory, snapshots = [], [], []
    for title in titles:
        number = title["number"]
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{number}.xml"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        payload = response.content
        root = ET.fromstring(payload)
        snapshot = DATA / f"title-{number}-{date}.xml"
        snapshot.write_bytes(payload)
        records, inventory = extract_units(root, number, date)
        # Reparse persisted source and require exact agreement for every reference.
        checked, checked_inventory = extract_units(ET.parse(snapshot).getroot(), number, date)
        if checked != records or checked_inventory != inventory:
            raise RuntimeError("Snapshot/reference verification failed")
        all_records.extend(records)
        all_inventory.extend(inventory)
        snapshots.append({"title": number, "name": title["name"], "url": url,
                          "file": snapshot.name, "bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest(),
                          "units": len(inventory), "questions": len(records)})
        print(f"Title {number}: {len(records)} questions, {len(inventory)} units", flush=True)
        time.sleep(1)
    question_file = DATA / "questions.jsonl"
    question_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                     for r in all_records), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "snapshot_date": date, "sources": snapshots,
                "question_count": len(all_records), "inventory": all_inventory,
                "question_file_sha256": hashlib.sha256(question_file.read_bytes()).hexdigest(),
                "scope": "All substantive SECTION and APPENDIX units in full Title 1/2 XML.",
                "coverage_limit": "One broad question per unit; not one question per legal fact. "
                "Reserved/empty units are inventoried without invented questions. "
                "References include all unit body text; external cross-references are not resolved.",
                "verification": "Every reference exactly matches its persisted XML unit after "
                "whitespace normalization. This proves text fidelity, not independent legal interpretation."}
    write_json(DATA / "manifest.json", manifest)
    print(f"Verified {len(all_records)} extractive references. Date: {date}", flush=True)


def load_verified():
    manifest = json.loads((DATA / "manifest.json").read_text())
    question_file = DATA / "questions.jsonl"
    if hashlib.sha256(question_file.read_bytes()).hexdigest() != manifest["question_file_sha256"]:
        raise RuntimeError("Question file changed after verification")
    records = [json.loads(line) for line in question_file.read_text().splitlines()]
    expected = []
    for source in manifest["sources"]:
        snapshot = DATA / source["file"]
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("Source snapshot hash mismatch")
        entries, _ = extract_units(ET.parse(snapshot).getroot(), source["title"], manifest["snapshot_date"])
        expected.extend(entries)
    if records != expected:
        raise RuntimeError("References no longer match source snapshot")
    return records, manifest


def run(args):
    import unsloth
    import torch
    from transformers import GenerationConfig
    from unsloth import FastLanguageModel

    records, manifest = load_verified()
    run_dir = DATA / "runs" / CONFIG["baseline_run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    responses_file = run_dir / "responses.jsonl"
    previous = [json.loads(line) for line in responses_file.read_text().splitlines()] if responses_file.exists() else []
    completed = {item["id"] for item in previous if item.get("status") == "ok"}
    pending = [r for r in records if r["id"] not in completed]
    if args.limit:
        pending = pending[:args.limit]
    print(f"Pending: {len(pending)}; already completed: {len(completed)}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(PATHS["model"]), max_seq_length=CONFIG["inference"]["context"],
        dtype=None, load_in_4bit=CONFIG["model"]["load_in_4bit"], full_finetuning=False, local_files_only=True,
        device_map={"": torch.cuda.current_device()},
    )
    FastLanguageModel.for_inference(model)
    model_revision = PATHS["revision"]
    write_json(run_dir / "settings.json", {
        "model": CONFIG["model"]["id"], "fine_tuned": False,
        "revision_record": model_revision.read_text() if model_revision.exists() else None,
        "snapshot_date": manifest["snapshot_date"], "seed": 3407,
        "max_new_tokens": args.max_new_tokens, "temperature": 0.7, "top_p": 0.8,
        "top_k": 20, "chat_template_kwargs": CONFIG["model"]["chat_template_kwargs"], "context": CONFIG["inference"]["context"],
        "system_prompt": None, "conversation": "fresh per question",
        "source_context_supplied": False,
    })
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=True, temperature=0.7,
        top_p=0.8, top_k=20, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    answers = ROOT / "Ai_answers.md"
    with answers.open("a", encoding="utf-8") as output:
        output.write(f"\n\n## Automated closed-book baseline — eCFR Titles 1 and 2\n\n"
                     f"Source snapshot: {manifest['snapshot_date']}. Configured base model; no adapter. "
                     "Each question uses a fresh conversation with no source text or system prompt. "
                     "Reference answers below are exact XML body text with normalized whitespace. "
                     "Model responses are unverified; token-overlap is not a correctness score.\n\n")
    for index, record in enumerate(pending, 1):
        if (DATA / "STOP_REQUESTED").exists():
            print("Stop requested. No additional question will be started.", flush=True)
            break
        torch.manual_seed(3407)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["question"]}], tokenize=False,
            add_generation_prompt=True, **CONFIG["model"]["chat_template_kwargs"],
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        started = time.monotonic()
        try:
            if inputs["input_ids"].shape[1] + args.max_new_tokens > CONFIG["inference"]["context"]:
                raise ValueError("Question exceeds context budget; not silently truncated")
            with torch.inference_mode():
                result = model.generate(**inputs, generation_config=generation_config)
            new_ids = result[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            status, error = "ok", None
            capped = len(new_ids) >= args.max_new_tokens and int(new_ids[-1]) != tokenizer.eos_token_id
        except Exception as exc:
            response, status, error, capped = "", "error", str(exc), False
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        item = {"id": record["id"], "question": record["question"],
                "response": response, "status": status, "error": error,
                "output_limit_reached": capped, "seconds": round(time.monotonic() - started, 2),
                "source_url": record["source_url"], "assessment": "not_manually_reviewed"}
        # Write human-readable record first, then mark completion for resumability.
        with answers.open("a", encoding="utf-8") as output:
            output.write(f"### {record['id']} — {record['heading']}\n\n"
                         f"**Question:** {record['question']}\n\n"
                         f"**Base model answer:**\n\n{response or error}\n\n"
                         f"**Output limit reached:** {capped}\n\n"
                         f"**Source:** {record['source_url']}\n\n"
                         f"**Verified reference answer (verbatim, whitespace normalized):**\n\n")
            for line in record["reference_answer"].splitlines():
                output.write(f"> {line}\n")
            output.write("\n**Assessment:** Pending manual factual review.\n\n")
        with responses_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
        if status == "ok":
            completed.add(record["id"])
        write_json(run_dir / "progress.json", {
            "total": len(records), "completed": len(completed),
            "last_id": record["id"], "last_status": status,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[{index}/{len(pending)}] {record['id']} {status} {item['seconds']}s", flush=True)
        if status != "ok":
            raise RuntimeError(f"Stopped after saving failed question: {error}")
    print("Requested baseline run completed.", flush=True)
    write_json(run_dir / "summary.json", {
        "total_questions": len(records), "completed": len(completed),
        "all_questions_completed": len(completed) == len(records),
        "reference_verification": "All references verified against hashed official XML snapshots.",
        "model_correctness": "Not automatically established; requires factual review.",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "run"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true",
                        help="Explicitly clear a stop marker and resume unanswered questions.")
    args = parser.parse_args()
    if args.limit < 0 or not 1 <= args.max_new_tokens <= 768:
        parser.error("Limit must be nonnegative; max-new-tokens must be between 1 and 768.")
    if args.mode == "prepare":
        prepare()
    elif args.mode == "verify":
        records, manifest = load_verified()
        print(f"Verified {len(records)} references against full Title 1/2 snapshots "
              f"for {manifest['snapshot_date']}.")
    else:
        import fcntl

        DATA.mkdir(parents=True, exist_ok=True)
        with (DATA / "run.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit("Another baseline process is already running.")
            stop_marker = DATA / "STOP_REQUESTED"
            if stop_marker.exists():
                if not args.resume:
                    raise SystemExit("Baseline intentionally stopped. Use --resume only when requested.")
                stop_marker.unlink()
            run(args)


if __name__ == "__main__":
    main()
