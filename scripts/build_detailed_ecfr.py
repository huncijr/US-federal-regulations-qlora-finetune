"""Build paragraph/sentence-level eCFR QA with source evidence and frozen source splits.

No language model generates target answers. Conservative extractive templates
preserve source lead-ins and explicit overriding exceptions. Complex/overlength
records are reported for further review instead of silently truncated.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from project_config import load_config, runtime_paths
from training_utils import check_split_overlap, load_examples

WORK = ROOT / "data" / "detailed_ecfr"
ENUM = re.compile(r"^((?:\([a-zA-Z0-9]+\)\s*)+)")


def normalized(value):
    return re.sub(r"\s+", " ", value).strip()


def element_text(element):
    return normalized("".join(element.itertext())) if element is not None else ""


def clean(value):
    value = ENUM.sub("", value).strip()
    value = re.sub(r";\s*(?:and|or)?$", ".", value)
    return value


def heading(element):
    value = element_text(element.find("HEAD"))
    value = re.sub(r"^(?:TITLE|CHAPTER|SUBCHAPTER|PART|SUBPART)\s+[^—–]+[—–]\s*", "", value, flags=re.I)
    value = re.sub(r"^§\s*[\d.]+\s*", "", value)
    if value.isupper():
        words = value.title().split()
        small = {"of", "the", "on", "and", "for", "to", "in", "a", "an", "or", "by", "with", "from", "at"}
        value = " ".join(word.lower() if index and word.lower() in small else word for index, word in enumerate(words))
    return value.rstrip(".")


def sentence_spans(value):
    # Mask abbreviations without changing character positions.
    masked = value
    for abbreviation in ("U.S.C.", "U.S.", "D.C.", "N.", "S.", "E.", "W.", "St.", "Ave.",
                         "Blvd.", "Rd.", "Ste.", "Apt.", "Mt.", "Ft.", "Jr.", "Sr.", "Ph.",
                         "e.g.", "i.e.", "No.", "Sec.", "Mr.", "Dr.", "Inc."):
        masked = masked.replace(abbreviation, abbreviation.replace(".", "\uFFF0"))
    boundaries = [0] + [match.end() for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z“])", masked)] + [len(value)]
    return [(start, end, value[start:end].strip()) for start, end in zip(boundaries, boundaries[1:])
            if value[start:end].strip()]


def paragraph_nodes(unit):
    nodes = []
    stack = []
    plain_lead = None
    for index, paragraph in enumerate(unit.iter("P")):
        value = element_text(paragraph)
        match = ENUM.match(value)
        markers = re.findall(r"\(([^)]+)\)", match.group(1)) if match else []
        if len(markers) > 1:
            levels = list(range(1, len(markers) + 1))
        elif markers:
            token = markers[0]
            if token.isdigit():
                level = 2
            elif token.isupper():
                level = 4
            elif re.fullmatch(r"[ivx]+", token) and any(entry[0] == 2 for entry in stack):
                level = 3
            else:
                level = 1
            levels = [level]
        else:
            levels = []
        if levels:
            first = levels[0]
            stack = [entry for entry in stack if entry[0] < first]
            ancestors = [entry[2] for entry in stack]
            if plain_lead is not None and not stack:
                ancestors = [plain_lead]
            for level, marker in zip(levels, markers):
                stack.append((level, marker, index))
            label = "".join(f"({entry[1]})" for entry in stack)
        else:
            ancestors = []
            label = ""
            stack = []
            if value.endswith((":", "—", "–")):
                plain_lead = index
            elif re.match(r".+?\s+(?:means|shall mean|refers to|includes)\b", value):
                plain_lead = None
        nodes.append({"index": index, "text": value, "label": label,
                      "ancestors": sorted(set(ancestors))})
    return nodes


def fetch_sources():
    import requests

    WORK.mkdir(parents=True, exist_ok=True)
    sources_dir = WORK / "sources"
    sources_dir.mkdir(exist_ok=True)
    session = requests.Session()
    response = session.get("https://www.ecfr.gov/api/versioner/v1/titles.json", timeout=90)
    response.raise_for_status()
    titles = [item for item in response.json()["titles"] if item["number"] in (1, 2)]
    date = min(item["up_to_date_as_of"] for item in titles)
    sources = []
    for item in titles:
        title = item["number"]
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
        response = session.get(url, timeout=120)
        response.raise_for_status()
        ET.fromstring(response.content)
        file = sources_dir / f"title-{title}-{date}.xml"
        file.write_bytes(response.content)
        sources.append({"title": title, "name": item["name"], "url": url,
                        "file": file.name, "sha256": hashlib.sha256(response.content).hexdigest()})
    manifest = {"date": date, "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": sources}
    (WORK / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def collect_units(manifest):
    units = []
    for source in manifest["sources"]:
        file = WORK / "sources" / source["file"]
        if hashlib.sha256(file.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("Source snapshot hash mismatch")
        root = ET.parse(file).getroot()

        def visit(element, path):
            kind = element.get("TYPE")
            if kind:
                path = path + [(kind, element.get("N"), heading(element))]
            if kind in ("SECTION", "APPENDIX"):
                title = source["title"]
                number = element.get("N")
                part = next((entry[1] for entry in reversed(path) if entry[0] == "PART"), "unknown")
                chapter = next((entry[2] for entry in path if entry[0] == "CHAPTER"), source["name"])
                part_heading = next((entry[2] for entry in reversed(path) if entry[0] == "PART"), "")
                topic = heading(element)
                reserved = bool(re.search(r"\[reserved\]", topic, re.I))
                scope = f"{chapter} rules on {part_heading}" if part_heading else f"{chapter} rules"
                if title == 1 and part in ("1", "2", "3"):
                    scope = "Federal Register publication rules"
                if title == 2 and part == "180":
                    scope = "federal nonprocurement debarment and suspension rules"
                citation = f"{title} CFR § {number}" if kind == "SECTION" else f"{title} CFR, {number}"
                url = f"https://www.ecfr.gov/on/{manifest['date']}/title-{title}/"
                url += f"section-{number}" if kind == "SECTION" else f"part-{part}"
                nodes = paragraph_nodes(element)
                overrides = defaultdict(list)
                for node in nodes:
                    refs = re.findall(r"[Nn]otwithstanding\s+paragraph\s+\(([a-z])\)", node["text"])
                    for ref in refs:
                        overrides[f"({ref})"].append(node["index"])
                units.append({"id": f"title-{title}:{kind.lower()}:{number}",
                              "title": title, "number": number, "kind": kind,
                              "part_key": f"title-{title}:part-{part}", "topic": topic,
                              "scope": scope, "citation": citation, "url": url,
                              "reserved": reserved, "nodes": nodes, "overrides": dict(overrides)})
                return
            for child in element:
                visit(child, path)
        visit(root, [])
    return units


def select_evidence(unit, node, descendants=False):
    indices = set(node["ancestors"] + [node["index"]])
    if descendants:
        indices.update(other["index"] for other in unit["nodes"] if node["index"] in other["ancestors"])
    for label, override in unit["overrides"].items():
        if node["label"].startswith(label):
            indices.update(override)
    # A short explicit chapter/part scope qualification belongs with definitions.
    if node["index"] and unit["nodes"]:
        first = unit["nodes"][0]
        if "unless the context requires otherwise" in first["text"].lower():
            indices.add(first["index"])
    return [unit["nodes"][i] for i in sorted(indices)]


def make_question(unit, node, sentence):
    value = clean(sentence)
    # Remove a small enumerated paragraph caption, not substantive sentences.
    if ENUM.match(sentence):
        value = re.sub(r"^[A-Za-z ]{2,35}\.\s+(?=[A-Z])", "", value)
    scope = unit["scope"]
    definition = re.match(r"^(.{2,100}?)\s+(?:shall mean|means|refers to|is defined as)\s+(.+)", value)
    if definition and not definition.group(1).lower().startswith(("this", "it", "which")):
        term = definition.group(1).strip('“”" ')
        return f"What does {term} mean in the {scope}?", "definition"
    inclusion = re.match(r"^(.{2,100}?)\s+(?:includes|include|consists of|contains)\s+(.+)", value)
    if inclusion:
        subject = re.sub(r"^(?:In addition,\s+)?", "", inclusion.group(1)).strip()
        subject = re.sub(r",?\s+(that|which|who)\s*$", "", subject).strip()
        if subject and subject[0].isupper():
            return f"What is included in {subject[0].lower() + subject[1:]} in the {scope}?", "contents"
    identity = re.match(r"^(.{3,100}?)\s+(is|are)\s+(.+)", value)
    if identity:
        subject, copula, predicate = identity.groups()
        subject = re.sub(r",?\s+(that|which|who)\s*$", "", subject).strip()
        if subject and subject[0].isupper() and subject.lower() not in ("this", "it", "which", "these", "there", "that"):
            if subject.split()[0] in ("Where", "When", "While", "Although", "Because", "Since", "Whether"):
                return None, None
            subject = subject[0].lower() + subject[1:] if subject.startswith(("The ", "An ", "A ", "Each ")) else subject
            if predicate.startswith("located"):
                if not re.search(r"located\s+(at|in|outside|on|http)", sentence, re.I):
                    return None, None
                return f"In the {scope}, where {copula} {subject} located?", "location"
            if re.search(r"\b(?:hours|price|cost|fee|amount|deadline|cutoff|frequency)\b", subject, re.I):
                return f"What {copula} {subject} in the {scope}?", "value_requirement"
            return f"How {copula} {subject} described in the {scope}?", "description"
    modal = re.match(r"^(.{3,100}?)\s+(must not|may not|shall not|must|shall|may|should|will)\s+(.+)", value)
    actor_clean = bool(modal) and not modal.group(1).lower().startswith(("if ", "when ", "unless ", "except "))
    actor_clean = actor_clean and modal.group(1).split()[0] not in ("On", "Upon", "After", "Before", "By", "For", "With", "Without", "In", "At", "Under", "During", "Through")
    if actor_clean:
        actor, verb, action = modal.groups()
        actor = actor[0].lower() + actor[1:] if actor.startswith(("The ", "An ", "Each ", "A ")) else actor
        focus = " ".join(action.split()[:9]).rstrip(".,;:")
        if "not" in verb:
            return f"In the {scope}, what restriction applies to {actor} concerning '{focus}'?", "restriction"
        if verb == "may":
            return f"In the {scope}, what is {actor} permitted to do concerning '{focus}', and subject to what conditions?", "permission"
        return f"In the {scope}, what obligation applies to {actor} concerning '{focus}'?", "obligation"
    timing = re.search(r"\b(within|no later than|at least|not more than|before|after)\b[^,;.]{0,80}?\b(days?|hours?|months?|years?)\b", value, re.I)
    if timing:
        phrase = normalized(timing.group(0))
        return f"For {unit['topic']} in the {scope}, what time requirement is stated as '{phrase}'?", "timing"
    if value.lower().startswith(("if ", "when ", "unless ", "except ", "notwithstanding ", "however")):
        anchor = " ".join(value.split()[:10]).rstrip(".,;:")
        return f"In the {scope}, what condition or exception is stated by the rule beginning '{anchor}'?", "condition_exception"
    if unit["topic"].endswith("?") and len(unit["nodes"]) <= 2:
        return f"Regarding the {scope}, {unit['topic'][0].lower() + unit['topic'][1:]}", "section_question"
    if node["ancestors"]:
        parent = unit["nodes"][node["ancestors"][-1]]
        parent_text = clean(parent["text"]).rstrip("—–:")
        parent_modal = re.match(r"^(.{3,100}?)\s+(must|shall|may|should|will)\s+(.+)", parent_text)
        if parent_modal:
            actor, _, parent_action = parent_modal.groups()
            actor = actor[0].lower() + actor[1:] if actor.startswith(("The ", "An ", "Each ", "A ")) else actor
            anchor = " ".join(value.split()[:8]).rstrip(".,;:")
            return f"In the {scope}, what detail concerning '{anchor}' is specified for {actor} to {' '.join(parent_action.split()[:6])}?", "procedure_detail"
    anchor = " ".join(value.split()[:10]).rstrip(".,;:")
    return f"For {unit['topic']} in the {scope}, what does the provision state concerning '{anchor}'?", "specific_source_fact"


def gold_examples(units):
    """Manually scoped, concise QA for the initial material and discovered failure cases."""
    mapping = {unit["id"]: unit for unit in units}
    rows = []

    def add(identifier, question, answer, evidence_texts, kind="reviewed_fact"):
        unit = mapping[identifier]
        indices = []
        for fragment in evidence_texts:
            matching = [node["index"] for node in unit["nodes"] if fragment in node["text"]]
            if not matching:
                raise RuntimeError(f"Reviewed fact not found in current source: {identifier}: {fragment}")
            indices.extend(matching)
        rows.append((unit, question, answer, sorted(set(indices)), kind))

    identifier = "title-1:section:2.2"
    add(identifier, "Who serves as Secretary of the Administrative Committee of the Federal Register?",
        "The Director of the Federal Register.", ["The Director of the Federal Register is the Secretary of the Committee."])
    add(identifier, "Which section of title 44 establishes the Administrative Committee of the Federal Register?",
        "Section 1506 of title 44, United States Code.", ["established by section 1506 of title 44"])
    add(identifier, "Who chairs the Administrative Committee of the Federal Register?",
        "The Archivist or Acting Archivist of the United States.", ["The Archivist, or Acting Archivist", "who is the Chairman"])
    add(identifier, "Who are the members of the Administrative Committee of the Federal Register?",
        "The Archivist or Acting Archivist of the United States, who is Chairman; an officer of the Department of Justice designated by the Attorney General; and the Public Printer or Acting Public Printer, using the wording of the regulation.",
        ["The Archivist, or Acting Archivist", "An officer of the Department of Justice", "The Public Printer or Acting Public Printer"])
    add(identifier, "Where should correspondence and materials intended for the Administrative Committee of the Federal Register be sent?",
        "To the Director of the Federal Register.", ["shall be sent to the Director of the Federal Register"])
    add(identifier, "Answer both questions separately: Who is Secretary of the Administrative Committee of the Federal Register, and which section of title 44 establishes it?",
        "1. The Director of the Federal Register is Secretary. 2. Section 1506 of title 44, United States Code, establishes the Committee.",
        ["is the Secretary of the Committee", "established by section 1506"], "reviewed_compound")
    add(identifier, "Someone says the Administrative Committee of the Federal Register is established by section 15 of title 44. Is that the correct section number?",
        "No. The correct section is 1506 of title 44. Chapter 15 and section 1506 are different citation levels.",
        ["established by section 1506 of title 44"], "reviewed_case")
    for identifier in ("title-1:section:457.151", "title-1:section:500.151"):
        scope = mapping[identifier]["scope"]
        add(identifier, f"In the {scope}, what accessibility requirement applies to new construction and alterations?",
            "Each building or building part constructed or altered by, on behalf of, or for the agency must be readily accessible to and usable by persons with disabilities. The regulation applies the Architectural Barriers Act, 42 U.S.C. 4151–4157, and the standards stated in 41 CFR 101-19.600 through 101-19.607.",
            ["Each building or part of a building", "42 U.S.C. 4151-4157"], "reviewed_fact")
        add(identifier, f"In the {scope}, which statute is cited for new-construction accessibility: the Architectural Barriers Act or Federal Acquisition Regulations Part 9?",
            "The provision cites the Architectural Barriers Act, 42 U.S.C. 4151–4157, with standards stated in 41 CFR 101-19.600 through 101-19.607. It does not cite FAR Part 9 for this requirement.",
            ["Architectural Barriers Act (42 U.S.C. 4151-4157)"], "reviewed_case")
    identifier = "title-1:section:426.201"
    add(identifier, "In 1 CFR § 426.201, which commission's own FOIA procedures are described?",
        "The National Commission on Military, National, and Public Service.", ["National Commission on Military, National, and Public Service"])
    add(identifier, "Which law governs access to records of the National Commission on Military, National, and Public Service in its FOIA procedures?",
        "The Freedom of Information Act, 5 U.S.C. 552, as amended.", ["Freedom of Information Act", "5 U.S.C. 552"])
    add(identifier, "I am requesting records about myself from the National Commission on Military, National, and Public Service. Are these requests processed only under FOIA?",
        "No. Requests for records about oneself under the Privacy Act of 1974, 5 U.S.C. 552a, are processed under the Commission's Privacy Act regulations as well as its FOIA procedures.",
        ["Requests made by individuals for records about themselves", "as well as under this subpart"], "reviewed_case")
    add(identifier, "Which additional guidelines should be read alongside the FOIA regulations of the National Commission on Military, National, and Public Service?",
        "The FOIA statute and OMB's Uniform Freedom of Information Fee Schedule and Guidelines.",
        ["Uniform Freedom of Information Fee Schedule and Guidelines", "text of the FOIA"])
    identifier = "title-2:section:180.215"
    add(identifier, "Under the federal nonprocurement debarment and suspension rules, which foreign-government-related entities can receive a direct award outside covered transactions?",
        "Paragraph (a) lists a foreign government or governmental entity; a public international organization; an entity owned wholly or partly, or controlled, by a foreign government; and other entities consisting wholly or partly of foreign governments or governmental entities. This direct-award exception is subject to paragraph (h)'s overriding arms-control, nonproliferation, and disarmament provisions.",
        ["A direct award to:", "A foreign government or foreign governmental entity", "A public international organization", "An entity owned (in whole or in part)", "Any other entity consisting wholly or partially", "Notwithstanding paragraph (a)"], "reviewed_exception")
    add(identifier, "Is every transaction with an entity owned partly by a foreign government automatically outside federal nonprocurement covered transactions?",
        "No. The stated exception concerns a direct award, not every transaction, and paragraph (h) can override the foreign-government-related exception.",
        ["A direct award to:", "An entity owned (in whole or in part)", "Notwithstanding paragraph (a)"], "reviewed_case")
    add(identifier, "A federal agency proposes a direct award to an entity partly owned by a foreign government, with no paragraph (h) circumstances. Is that transaction listed among the exceptions to nonprocurement covered transactions?",
        "Yes. A direct award to an entity owned wholly or partly, or controlled, by a foreign government is listed in paragraph (a)(3). The answer assumes paragraph (h) does not override the exception.",
        ["A direct award to:", "An entity owned (in whole or in part)", "Notwithstanding paragraph (a)"], "reviewed_case")
    add(identifier, "Can foreign-government ownership alone justify an award to an entity listed in SAM.gov Exclusions for the arms-control-related activities described in the overriding rule?",
        "No. Paragraph (h) overrides paragraph (a) and prohibits the specified awards, renewals, or extensions to entities listed on that basis. The stated exception requires the Federal agency head's action under 2 CFR 180.135 and concurrence of the OMB Director.",
        ["Notwithstanding paragraph (a)", "must not award, renew, or extend", "with the concurrence of the OMB Director"], "reviewed_case")
    add(identifier, "Does the foreign-government direct-award exception depend on whether the transaction occurs on a covered Federal computer system?",
        "No such computer-system condition appears in this provision. The relevant conditions are the direct-award classification and recipient categories in paragraph (a), subject to the overriding paragraph (h).",
        ["A direct award to:", "Notwithstanding paragraph (a)"], "reviewed_case")
    return rows


def build(manifest, tokenizer, config):
    units = collect_units(manifest)
    unit_by_id = {unit["id"]: unit for unit in units}
    accepted, deferred, seen, inventory = [], [], set(), []
    limit = config["training"]["max_seq_length"]
    template_kwargs = config["model"]["chat_template_kwargs"]

    def emit(unit, question, answer, evidence, kind, method):
        question = normalized(question)
        if question in seen:
            return
        seen.add(question)
        quotes = [unit["nodes"][index]["text"] for index in evidence]
        if not quotes or any(not quote for quote in quotes):
            return
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **template_kwargs)
        tokens = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        record = {"messages": messages, "metadata": {
            "source_id": unit["id"], "citation": unit["citation"], "source_url": unit["url"],
            "snapshot_date": manifest["date"], "source_part": unit["part_key"],
            "source_paragraph_indices": evidence, "supporting_quotes": quotes,
            "support_sha256": hashlib.sha256("\n".join(quotes).encode()).hexdigest(),
            "question_type": kind, "curation_method": method, "tokens": tokens,
            "answer_verification": "source_checked_focused_answer" if method == "reviewed" else "complete_source_quotes_with_lead_ins",
            "semantic_review": "reviewed_against_source" if method == "reviewed" else "deterministic_extractive_template_not_independent_legal_certification",
        }}
        (accepted if tokens <= limit else deferred).append(record)

    for unit, q, a, indices, kind in gold_examples(units):
        emit(unit, q, a, indices, kind, "reviewed")

    # Retain the original manually prepared initial examples, with explicit citation scope.
    original = ROOT / "data" / "all_examples.jsonl"
    if original.exists():
        for record in load_examples(original)[:25]:
            match = re.match(r"1 CFR § ([\d.]+)", record["metadata"].get("citation", ""))
            if not match:
                continue
            unit = unit_by_id.get("title-1:section:" + match.group(1))
            if unit:
                # Evidence is the complete short source section for these initial examples.
                emit(unit, record["messages"][0]["content"], record["messages"][-1]["content"],
                     [node["index"] for node in unit["nodes"]], "initial_reviewed_fact", "reviewed")

    for unit in units:
        start = len(accepted)
        deferred_start = len(deferred)
        if not unit["reserved"]:
            for node in unit["nodes"]:
                if len(clean(node["text"]).split()) < 4:
                    continue
                has_children = any(node["index"] in other["ancestors"] for other in unit["nodes"])
                if node["text"].endswith((":", "—", "–")) and not has_children:
                    continue
                spans = sentence_spans(node["text"]) if not has_children else [(0, len(node["text"]), node["text"])]
                for sentence_index, (_, _, sentence) in enumerate(spans):
                    if len(clean(sentence).split()) < 5:
                        continue
                    question, kind = make_question(unit, node, sentence)
                    if not question:
                        continue
                    evidence = select_evidence(unit, node, descendants=has_children)
                    # Keep complete paragraph/child context for the target, not a clipped sentence.
                    answer = "\n".join(clean(item["text"]).rstrip("—–:") for item in evidence)
                    if len(spans) > 1:
                        # Focused answers use a complete source sentence plus any required lead-ins.
                        prefix = [item for item in evidence if item["index"] != node["index"]]
                        if sentence_index and sentence.lower().startswith(("however", "but ", "this ", "these ", "it ")):
                            answer = "\n".join(clean(item["text"]) for item in prefix + [node])
                        else:
                            answer = "\n".join([clean(item["text"]).rstrip("—–:") for item in prefix] + [clean(sentence)])
                    if question in seen:
                        stem = question.rstrip("?")
                        question = f"{stem} (specifically {unit['citation']}{node['label']}, statement {sentence_index + 1})?"
                    emit(unit, question, answer, [item["index"] for item in evidence], kind, "extractive")
        inventory.append({"id": unit["id"], "part": unit["part_key"], "reserved": unit["reserved"],
                          "paragraphs": len(unit["nodes"]), "new_usable_examples": len(accepted) - start,
                          "new_overlength_examples": len(deferred) - deferred_start})
    # Recheck every evidence quote against the parsed official source.
    for record in accepted + deferred:
        meta = record["metadata"]
        unit = unit_by_id[meta["source_id"]]
        quotes = [unit["nodes"][i]["text"] for i in meta["source_paragraph_indices"]]
        if quotes != meta["supporting_quotes"]:
            raise RuntimeError("Evidence quote mismatch")
        if hashlib.sha256("\n".join(quotes).encode()).hexdigest() != meta["support_sha256"]:
            raise RuntimeError("Evidence hash mismatch")
    return accepted, deferred, inventory


def assign_splits(records, old_manifest):
    fixed = {part: split for split, info in old_manifest["splits"].items() for part in info["source_parts"]}
    assignments = dict(fixed)
    for part in sorted({r["metadata"]["source_part"] for r in records}):
        if part not in assignments:
            value = int(hashlib.sha256(("3407:" + part).encode()).hexdigest()[:8], 16) % 100
            # The general award guidance is foundational training material, not a new holdout.
            assignments[part] = "train" if part == "title-2:part-200" or value < 80 else "validation" if value < 90 else "test"
    splits = {name: [] for name in ("train", "validation", "test")}
    # Exact substantial answer duplicates must not leak into a lower-priority split.
    priority = {"test": 0, "validation": 1, "train": 2}
    answer_owner = {}
    for record in records:
        answer = normalized(record["messages"][-1]["content"]).lower()
        if len(answer.split()) < 12:
            continue
        split = assignments[record["metadata"]["source_part"]]
        previous = answer_owner.get(answer)
        if previous is None or priority[split] < priority[previous]:
            answer_owner[answer] = split
    removed = []
    for record in records:
        split = assignments[record["metadata"]["source_part"]]
        answer = normalized(record["messages"][-1]["content"]).lower()
        if answer in answer_owner and answer_owner[answer] != split:
            removed.append(record)
        else:
            splits[split].append(record)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        check_split_overlap(splits[left], splits[right])
    return splits, assignments, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reuse-sources", action="store_true")
    args = parser.parse_args()
    config = load_config(ROOT)
    paths = runtime_paths(config, ROOT)
    source_manifest = WORK / "source_manifest.json"
    manifest = json.loads(source_manifest.read_text()) if args.reuse_sources and source_manifest.exists() else fetch_sources()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], local_files_only=True)
    records, deferred, inventory = build(manifest, tokenizer, config)
    old_manifest = json.loads((ROOT / "data" / "split_manifest.json").read_text())
    splits, assignments, removed = assign_splits(records, old_manifest)

    def write_rows(file, rows):
        file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    write_rows(WORK / "catalog.jsonl", records)
    write_rows(WORK / "overlength_review.jsonl", deferred)
    write_rows(WORK / "cross_split_duplicates.jsonl", removed)
    report = {
        "snapshot_date": manifest["date"], "usable_catalog_examples": len(records),
        "overlength_deferred": len(deferred), "cross_split_duplicates_removed": len(removed),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "question_types": dict(Counter(r["metadata"]["question_type"] for r in records)),
        "max_tokens": max(r["metadata"]["tokens"] for r in records),
        "source_parts": assignments, "unit_inventory": inventory,
        "review_limit": "Evidence is verified for every example. Templates are extractive, not a claim of independent legal interpretation. Overlength complete contexts are deferred, never truncated.",
        "training_started": False,
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("unit_inventory", "source_parts")}, indent=2))
    if not args.apply:
        print("Catalog prepared; active train/validation/test files have not been changed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = ROOT / "data" / "history" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for file in [paths[name] for name in ("train", "validation", "test")] + [ROOT / "data" / "all_examples.jsonl", ROOT / "data" / "split_manifest.json"]:
        shutil.copy2(file, backup / file.name)
    for name, rows in splits.items():
        write_rows(paths[name], rows)
        load_examples(paths[name])
    write_rows(ROOT / "data" / "all_examples.jsonl", records)
    updated = {
        "version": "detailed_ecfr_v2", "seed": 3407, "total_examples": sum(map(len, splits.values())),
        "catalog_examples": len(records), "grouping": "frozen historical CFR parts; deterministic new-part allocation; exact substantial-answer deduplication across splits",
        "source_snapshot": manifest["date"], "splits": {
            name: {"count": len(rows), "source_parts": sorted({r["metadata"]["source_part"] for r in rows}),
                   "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest()}
            for name, rows in splits.items()},
        "previous_dataset_backup": str(backup.relative_to(ROOT)),
        "evidence_verified": True, "semantic_review_limit": report["review_limit"],
    }
    (ROOT / "data" / "split_manifest.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
    print("Applied detailed datasets. Previous files preserved at", backup.relative_to(ROOT))
    print("Approximate optimizer updates for one epoch:", math.ceil(len(splits["train"]) / config["training"]["gradient_accumulation_steps"]))


if __name__ == "__main__":
    main()
