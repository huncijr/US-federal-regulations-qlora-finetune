"""Create focused QA drafts from completed baseline source paragraphs, not model answers."""

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecfr_baseline import DATA, ROOT, load_verified, text, write_json
from training_utils import load_examples
from project_config import load_config, runtime_paths

WORK = DATA / "curation"
CONFIG = load_config(ROOT)
PATHS = runtime_paths(CONFIG, ROOT)
TARGET = PATHS["train"]

# Reviewed replacements for questions whose drafts ask for facts not in the quote.
QUESTION_CORRECTIONS = {
    "title-1:section:3.2": "Under 1 CFR § 3.2, where and during what hours are filed documents available for public inspection?",
    "title-1:section:11.1": "Under 1 CFR § 11.1, who makes Federal Register publications available for sale, and to whom are fees payable in advance?",
    "title-1:section:10.2": "Under 1 CFR § 10.2, which Presidential statements and document categories form the content of the publication?",
    "title-1:section:19.5": "Under 1 CFR § 19.5, which international instruments are outside the scope of these regulations?",
    "title-1:section:304.10": "Under 1 CFR § 304.10, when may requested records be disposed of, and may disposal occur during a pending FOIA request, appeal, or lawsuit?",
    "title-1:section:457.130": "Under 1 CFR § 457.130(a), what discrimination against qualified handicapped persons is prohibited in agency programs or activities?",
    "title-1:section:500.130": "Under 1 CFR § 500.130(a), what discrimination against qualified handicapped persons is prohibited in agency programs or activities?",
    "title-1:section:601.5": "Under 1 CFR § 601.5(a), which NEPA role must a Federal Agency applicant designated as Lead Agency perform?",
    "title-1:section:601.16": "Under 1 CFR § 601.16(a), when is a FONSI prepared and signed if NCPC is the Lead Agency?",
    "title-1:section:602.4": "Under 1 CFR § 602.4, where must NCPC make Reading Room Materials available electronically and physically?",
    "title-2:section:25.105": "Under 2 CFR § 25.105, to which applicants and recipients does Part 25 apply, and what exclusions apply to individuals and lower-tier entities?",
    "title-2:section:170.210": "Under 2 CFR § 170.210, what compliance processes and systems must applicants have in place if they receive a Federal award?",
    "title-2:section:175.200": "Under 2 CFR § 175.200, for which grants or cooperative agreements must a Federal agency include the award term in Appendix A?",
    "title-2:section:176.150": "Under 2 CFR § 176.150(a), which provision defines manufactured good, public building and public work, and steel for this notice?",
    "title-2:section:176.170": "Under 2 CFR § 176.170(a), which provision supplies the definitions used in this award provision?",
    "title-2:section:176.160": "Under 2 CFR § 176.160(a), which categories of countries are included in the definition of a designated country?",
    "title-2:appendix:Appendix to Subpart B of 2 CFR Part 176": "Under 2 CFR, the Appendix to Subpart B of Part 176, which types of projects are covered by the listed restrictions on Federal funds to States?",
    "title-2:section:180.130": "Under 2 CFR § 180.130, what participation and principal roles are prohibited for an excluded person, subject to the stated exceptions?",
    "title-2:section:180.220": "Under 2 CFR § 180.220, when can a procurement contract awarded under a nonprocurement transaction be a covered transaction?",
}


def token_count(tokenizer, question, answer):
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": answer}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **CONFIG["model"]["chat_template_kwargs"],
    )
    return len(tokenizer(formatted, add_special_tokens=False)["input_ids"])


def prepare():
    from transformers import AutoTokenizer

    records, manifest = load_verified()
    completed = {json.loads(line)["id"] for line in
                 (DATA / "completed_questions.jsonl").read_text().splitlines()}
    existing = load_examples(TARGET)
    covered = {match.groups() for e in existing
               if (match := re.match(r"(\d+) CFR § ([\d.]+)", e.get("metadata", {}).get("citation", "")))}
    covered_ids = {e.get("metadata", {}).get("source_id") for e in existing}
    tokenizer = AutoTokenizer.from_pretrained(PATHS["model"], local_files_only=True)
    trees = {source["title"]: ET.parse(DATA / source["file"]).getroot()
             for source in manifest["sources"]}
    candidates = []
    for record in records:
        if record["id"] not in completed:
            continue
        if record["id"] in covered_ids or (str(record["title"]), record["number"]) in covered:
            continue
        unit = next(e for e in trees[record["title"]].iter()
                    if e.get("TYPE") == record["kind"] and e.get("N") == record["number"])
        # Select a complete substantive paragraph, never a token-truncated excerpt.
        paragraphs = [(i, p) for i, p in enumerate(unit.iter("P"))
                      if text(p) and not text(p).endswith((":", "—", "–"))]
        options = []
        for index, paragraph in paragraphs:
            quote = text(paragraph)
            if len(quote.split()) < 6:
                continue
            n = len(tokenizer(quote, add_special_tokens=False)["input_ids"])
            if n <= 340:
                options.append((index, quote, n))
        full_length = len(tokenizer(record["reference_answer"], add_special_tokens=False)["input_ids"])
        if full_length <= 340:
            quote = record["reference_answer"]
            selected = None
        elif not options:
            # Preserve a full answer for a longer-sequence review rather than inventing text.
            quote = record["reference_answer"]
            selected = None
        else:
            selected, quote, _ = options[0]
            # List items inherit their lead-in, including negations and conditions.
            # Retain a contiguous source span rather than quoting an orphaned item.
            raw_paragraphs = [text(p) for p in unit.iter("P")]
            start = selected
            end = selected
            if re.match(r"^\((?:\d+|[ivx]+)\)", quote):
                for previous in range(selected - 1, -1, -1):
                    if raw_paragraphs[previous].endswith((":", "—", "–")):
                        start = previous
                        break
            while end + 1 < len(raw_paragraphs) and raw_paragraphs[end].endswith((";", " or", " and", "—", ":")):
                end += 1
            span = "\n\n".join(raw_paragraphs[start:end + 1])
            if start != selected or end != selected:
                span_tokens = len(tokenizer(span, add_special_tokens=False)["input_ids"])
                if span_tokens <= 600:
                    quote = span
                else:
                    alternatives = [(i, q) for i, q, _ in options
                                    if not re.match(r"^\((?:\d+|[ivx]+)\)", q)
                                    and not q.endswith((";", " or", " and"))]
                    if not alternatives:
                        raise RuntimeError(f"Needs manual complete-context extraction: {record['id']}")
                    selected, quote = alternatives[0]
        candidates.append({**record, "answer_quote": quote, "paragraph_index": selected,
                           "answer_verification": "exact_normalized_source_span" if selected is not None
                           else "exact_normalized_section_body",
                           "quote_sha256": hashlib.sha256(quote.encode()).hexdigest()})
    WORK.mkdir(parents=True, exist_ok=True)
    backup = WORK / "original_25_examples.jsonl"
    if not backup.exists():
        backup.write_bytes(TARGET.read_bytes())
    (WORK / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates), encoding="utf-8",
    )
    print(f"Candidates: {len(candidates)}; existing examples: {len(existing)}", flush=True)
    print("Full-section references:", sum(r["paragraph_index"] is None for r in candidates))


def generate(limit=0):
    import unsloth
    import torch
    from unsloth import FastLanguageModel

    candidates = [json.loads(line) for line in (WORK / "candidates.jsonl").read_text().splitlines()]
    drafts_file = WORK / "drafts.jsonl"
    previous = [json.loads(line) for line in drafts_file.read_text().splitlines()] if drafts_file.exists() else []
    expected_quotes = {row["id"]: row["answer_quote"] for row in candidates}
    previous = [row for row in previous if
                row["example"]["messages"][1]["content"] == expected_quotes.get(row["source_id"])]
    drafts_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in previous), encoding="utf-8")
    done = {row["source_id"] for row in previous}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(PATHS["model"]), max_seq_length=1024,
        dtype=None, load_in_4bit=CONFIG["model"]["load_in_4bit"], full_finetuning=False, local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    generated = 0
    for index, candidate in enumerate(candidates, 1):
        if candidate["id"] in done:
            continue
        quote = candidate["answer_quote"]
        citation = (f"{candidate['title']} CFR § {candidate['number']}" if candidate["kind"] == "SECTION"
                    else f"{candidate['title']} CFR, {candidate['number']}")
        instruction = (
            "Write ONE specific English factual question answered by the regulatory excerpt below. "
            "Ask about a concrete definition, actor, requirement, condition, deadline, procedure, "
            "exception, or location in the excerpt. Do not ask to summarize the section. "
            "Do not provide an answer, explanations, a list, or quotation marks. "
            "Do not assume facts absent from the excerpt. End with a question mark. "
            "The citation will be added separately.\n\n"
            f"Section heading: {candidate['heading']}\nExcerpt:\n{quote}"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False,
            add_generation_prompt=True, **CONFIG["model"]["chat_template_kwargs"],
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if inputs["input_ids"].shape[1] + 96 > 1024:
            raise RuntimeError(f"Question-generation context too long for {candidate['id']}")
        torch.manual_seed(3407)
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=96, do_sample=True,
                                    temperature=0.5, top_p=0.9, top_k=20)
        draft = tokenizer.decode(result[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        draft = re.sub(r"^(?:Question\s*:\s*|\d+[.)]\s*|[-*]\s*)", "", draft, flags=re.I)
        draft = draft.strip('"')
        if "?" not in draft:
            raise RuntimeError(f"Incomplete question for {candidate['id']}: {draft}")
        draft = draft.split("?", 1)[0].strip() + "?"
        if any(term in draft.lower() for term in ("what does this provision state", "summarize this section", "what does this section")):
            raise RuntimeError(f"Non-specific question for {candidate['id']}: {draft}")
        question = f"Under {citation}, {draft[0].lower() + draft[1:]}"
        example = {
            "messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": quote}],
            "metadata": {
                "citation": citation, "source_url": candidate["source_url"],
                "snapshot_date": candidate["snapshot_date"], "source_id": candidate["id"],
                "source_paragraph_index": candidate["paragraph_index"],
                "answer_verification": candidate["answer_verification"],
                "answer_sha256": candidate["quote_sha256"],
                "question_method": "source_conditioned_local_model_draft",
                "question_review": "requires_semantic_review",
                "source_heading": candidate["heading"],
                "tokens": token_count(tokenizer, question, quote),
            },
        }
        with drafts_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"source_id": candidate["id"], "example": example}, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(candidates)}] {question}", flush=True)
        generated += 1
        if limit and generated >= limit:
            break


def merge():
    # Re-verify the official source snapshots before accepting any generated draft.
    records, _ = load_verified()
    source = {row["id"]: row for row in records}
    candidates = {r["id"]: r for r in map(json.loads, (WORK / "candidates.jsonl").read_text().splitlines())}
    drafts = list(map(json.loads, (WORK / "drafts.jsonl").read_text().splitlines()))
    if set(candidates) != {row["source_id"] for row in drafts}:
        raise RuntimeError("Not all completed source units have generated questions yet")
    originals = [row for row in load_examples(TARGET)
                 if row.get("metadata", {}).get("source_id") not in candidates]
    examples = list(originals)
    seen = {row["messages"][0]["content"] for row in examples}
    for draft in drafts:
        row = draft["example"]
        if draft["source_id"] in QUESTION_CORRECTIONS:
            row["messages"][0]["content"] = QUESTION_CORRECTIONS[draft["source_id"]]
            row["metadata"]["question_review"] = "targeted_question_manually_corrected_against_source"
        answer = row["messages"][1]["content"]
        expected = candidates[draft["source_id"]]
        row["metadata"]["source_paragraph_index"] = expected["paragraph_index"]
        row["metadata"]["answer_verification"] = expected["answer_verification"]
        if expected["kind"] == "APPENDIX":
            row["metadata"]["citation"] = f"{expected['title']} CFR, {expected['number']}"
            row["messages"][0]["content"] = row["messages"][0]["content"].replace(
                f"{expected['title']} CFR § {expected['number']}", row["metadata"]["citation"])
        normalized_source = re.sub(r"\s+", " ", source[draft["source_id"]]["reference_answer"])
        if answer != expected["answer_quote"] or re.sub(r"\s+", " ", answer) not in normalized_source:
            raise RuntimeError(f"Answer differs from verified source: {draft['source_id']}")
        if hashlib.sha256(answer.encode()).hexdigest() != row["metadata"]["answer_sha256"]:
            raise RuntimeError("Answer hash mismatch")
        if row["messages"][0]["content"] in seen:
            raise RuntimeError("Duplicate question")
        seen.add(row["messages"][0]["content"])
        examples.append(row)
    TARGET.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8")
    load_examples(TARGET)
    write_json(WORK / "report.json", {
        "preserved_examples": len(originals), "added_examples": len(drafts),
        "total_examples": len(examples), "new_answers_verified": len(drafts),
        "over_512_tokens": sum(row["example"]["metadata"]["tokens"] > 512 for row in drafts),
        "question_review": "Draft questions require semantic review; source equality does not prove question-answer fit.",
        "coverage": "At least one selected fact per completed source unit; not every fact or paragraph.",
        "training_started": False,
    })
    print(f"Merged {len(examples)} examples; {len(drafts)} new answers verified against official source.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate", "merge"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if (ROOT / "data" / "split_manifest.json").exists():
        raise SystemExit(
            "The dataset is already split. Curation must not modify the training file "
            "or pull held-out source units back into training. Use a separately curated "
            "new dataset and intentionally revise the split before running this tool."
        )
    if args.mode == "generate":
        generate(args.limit)
    else:
        {"prepare": prepare, "merge": merge}[args.mode]()
