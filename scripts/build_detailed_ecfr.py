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


def seen_key(question):
    return normalized(question).lower()


CUT_END = re.compile(r",\s*(and|or)\s*$|[:;]\s*$")


def extend_cut_evidence(unit, evidence):
    """Append following plain-paragraph nodes while evidence ends mid-thought.

    Returns the extended index list, or None when the thought cannot be
    completed from subsequent plain paragraphs.
    """
    evidence = list(evidence)
    guard = 0
    while guard < 3:
        texts = [unit["nodes"][i]["text"] for i in evidence]
        if not texts or not CUT_END.search(texts[-1].strip()):
            break
        nxt = evidence[-1] + 1
        if nxt >= len(unit["nodes"]) or unit["nodes"][nxt]["tag"] not in ("P", "FP", "FP-1", "FP-DASH", "PSPACE"):
            return None
        evidence.append(nxt)
        guard += 1
    final_texts = [unit["nodes"][i]["text"] for i in evidence]
    if final_texts and CUT_END.search(final_texts[-1].strip()):
        return None
    return sorted(set(evidence))


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
                         "Ch.", "Pt.", "Art.", "Reg.", "Vol.", "Dept.", "F.R.",
                         "e.g.", "i.e.", "No.", "Sec.", "Mr.", "Dr.", "Inc."):
        masked = masked.replace(abbreviation, abbreviation.replace(".", "\uFFF0"))
    boundaries = [0] + [match.end() for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z“])", masked)] + [len(value)]
    return [(start, end, value[start:end].strip()) for start, end in zip(boundaries, boundaries[1:])
            if value[start:end].strip()]


def paragraph_nodes(unit):
    nodes = []
    stack = []
    plain_lead = None
    parents = {child: parent for parent in unit.iter() for child in parent}
    blocks = []
    formatted = set()
    for element in unit.iter():
        if element.tag not in ("P", "FP", "FP-1", "FP-DASH", "PSPACE"):
            continue
        ancestor = parents.get(element)
        excluded = False
        while ancestor is not None and ancestor is not unit:
            if ancestor.tag == "EXTRACT":
                formatted.add(element)
            if ancestor.tag in ("TABLE", "TD", "TH", "CITA", "SOURCE", "FTNT"):
                excluded = True
                break
            ancestor = parents.get(ancestor)
        if not excluded:
            blocks.append(element)
    for index, paragraph in enumerate(blocks):
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
            if paragraph.tag != "P" or paragraph in formatted:
                ancestors = [entry[2] for entry in stack]
                if plain_lead is not None:
                    ancestors.append(plain_lead)
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
                      "ancestors": sorted(set(ancestors)), "tag": paragraph.tag})
    return nodes


def parse_titles(value):
    selected = set()
    try:
        for piece in value.split(","):
            if "-" in piece:
                start, end = map(int, piece.split("-", 1))
                if start > end:
                    raise ValueError()
                selected.update(range(start, end + 1))
            else:
                selected.add(int(piece))
    except ValueError:
        raise argparse.ArgumentTypeError("Use title numbers/ranges such as 1-10 or 1,2,3-7.")
    if not selected or not all(1 <= number <= 50 for number in selected):
        raise argparse.ArgumentTypeError("Title numbers must be between 1 and 50.")
    return sorted(selected)


def fetch_sources(requested_titles):
    import requests

    WORK.mkdir(parents=True, exist_ok=True)
    sources_dir = WORK / "sources"
    sources_dir.mkdir(exist_ok=True)
    session = requests.Session()
    response = session.get("https://www.ecfr.gov/api/versioner/v1/titles.json", timeout=90)
    response.raise_for_status()
    available = [item for item in response.json()["titles"] if item["number"] in requested_titles]
    reserved = [item["number"] for item in available if item.get("reserved")]
    titles = [item for item in available if not item.get("reserved")]
    if not titles:
        raise RuntimeError("Requested titles contain no active source text.")
    date = min(item["up_to_date_as_of"] for item in titles)
    prior_manifest_file = WORK / "source_manifest.json"
    prior = json.loads(prior_manifest_file.read_text()) if prior_manifest_file.exists() else {}
    prior_hashes = {source["file"]: source["sha256"] for source in prior.get("sources", [])}
    cache_index_file = sources_dir / "cache_index.json"
    if cache_index_file.exists():
        prior_hashes.update(json.loads(cache_index_file.read_text()))
    sources = []
    for item in titles:
        title = item["number"]
        url = f"https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
        file = sources_dir / f"title-{title}-{date}.xml"
        if file.exists():
            payload = file.read_bytes()
            if file.name not in prior_hashes or hashlib.sha256(payload).hexdigest() != prior_hashes[file.name]:
                raise RuntimeError(f"Unverified or changed cached source: {file.name}")
            ET.fromstring(payload)
        else:
            print(f"Downloading official Title {title} snapshot ({date})...", flush=True)
            response = session.get(url, timeout=180)
            response.raise_for_status()
            payload = response.content
            ET.fromstring(payload)
            file.write_bytes(payload)
        sources.append({"title": title, "name": item["name"], "url": url,
                        "file": file.name, "sha256": hashlib.sha256(payload).hexdigest()})
        prior_hashes[file.name] = sources[-1]["sha256"]
        cache_index_file.write_text(json.dumps(prior_hashes, indent=2), encoding="utf-8")
    manifest = {"date": date, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "requested_titles": requested_titles, "reserved_titles": reserved, "sources": sources}
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
                if nodes and all(node["text"].strip().lower() == "[reserved]" for node in nodes):
                    reserved = True
                tables = []
                for table_index, table in enumerate(element.iter("TABLE")):
                    if any(int(cell.get("colspan", "1")) > 1 or int(cell.get("rowspan", "1")) > 1 for cell in table.iter() if cell.tag in ("TD", "TH")):
                        tables.append({"index": table_index, "complex": True, "rows": []})
                        continue
                    headers = []
                    for cell in table.findall("./THEAD/TR/TH"):
                        index = len(nodes)
                        nodes.append({"index": index, "text": element_text(cell), "label": "", "ancestors": [], "tag": "TH"})
                        headers.append(index)
                    rows = []
                    for row in table.findall("./TBODY/TR"):
                        indices = []
                        for cell in row.findall("TD"):
                            index = len(nodes)
                            nodes.append({"index": index, "text": element_text(cell), "label": "", "ancestors": [], "tag": "TD"})
                            indices.append(index)
                        rows.append(indices)
                    tables.append({"index": table_index, "complex": False, "headers": headers, "rows": rows})
                overrides = defaultdict(list)
                for node in nodes:
                    refs = re.findall(r"[Nn]otwithstanding\s+paragraph\s+\(([a-z])\)", node["text"])
                    for ref in refs:
                        overrides[f"({ref})"].append(node["index"])
                units.append({"id": f"title-{title}:{kind.lower()}:{number}",
                              "title": title, "number": number, "kind": kind,
                              "part_key": f"title-{title}:part-{part}", "topic": topic,
                              "scope": scope, "citation": citation, "url": url,
                              "reserved": reserved, "nodes": nodes, "overrides": dict(overrides),
                              "tables": tables, "images": len(list(element.iter("img")))})
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
        value = re.sub(r"^[A-Za-z ]{2,35}[—–]\s*(?:\([a-zA-Z0-9]+\)\s*)+[A-Za-z ]{2,35}\.\s+", "", value)
    scope = unit["scope"]
    definition = re.match(r"^(.{2,100}?)\s+(?:shall mean|means|refers to|is defined as)\s+(.+)", value)
    if definition and not definition.group(1).lower().startswith(("this", "it", "which")) and not re.search(r"\b(shall|must|may|will|should)\b", definition.group(1), re.I):
        term = definition.group(1).strip('“”" ')
        if term.split()[0] not in ("If", "In", "On", "For", "As", "When", "With", "By", "Under", "From", "To"):
            return f"What does {term} mean in the {scope}?", "definition"
    inclusion = re.match(r"^(.{2,100}?)\s+(?:includes|include|consists of|contains)\s+(.+)", value)
    if inclusion:
        subject = re.sub(r"^(?:In addition,\s+)?", "", inclusion.group(1)).strip()
        subject = re.sub(r",?\s+(that|which|who)\s*$", "", subject).strip()
        if subject and subject[0].isupper() and subject.lower() not in ("this", "it", "which", "these", "there", "that", "these data"):
            if re.search(r"\b(is|are|was|were|do not|does not|did not)\b", subject):
                return None, None
            return f"What is included in {subject[0].lower() + subject[1:]} in the {scope}?", "contents"
    identity = re.match(r"^(.{3,100}?)\s+(is|are)\s+(.+)", value)
    if identity:
        subject, copula, predicate = identity.groups()
        subject = re.sub(r",?\s+(that|which|who)\s*$", "", subject).strip()
        invalid_subject = not subject or bool(re.search(r"\b(shall|must|may|will|should)\b", subject, re.I)) or subject.split()[0] in ("If", "In", "On", "Upon", "Before", "After", "Provided", "Except", "Notwithstanding", "With", "Without", "During", "As", "Where", "When", "While", "Until", "Although", "Because", "Since", "Whether")
        if subject and subject[0].isupper() and not invalid_subject and subject.lower() not in ("this", "it", "which", "these", "there", "that"):
            subject = subject[0].lower() + subject[1:] if subject.startswith(("The ", "An ", "A ", "Each ")) else subject
            if predicate.startswith("located"):
                if not re.search(r"located\s+(at|in|outside|on|http)", sentence, re.I):
                    return None, None
                return f"In the {scope}, where {copula} {subject} located?", "location"
            if re.search(r"\b(?:hours|price|cost|fee|amount|deadline|cutoff|frequency)\b", subject, re.I):
                return f"What {copula} {subject} in the {scope}?", "value_requirement"
            return f"How {copula} {subject} described in the {scope}?", "description"
    modal = re.match(r"^(.{3,100}?)\s+(must not|may not|shall not|will not|should not|must|shall|may|should|will)(?:\s*,\s*|\s+)(.+)", value)
    actor_clean = bool(modal) and not modal.group(1).lower().startswith(("if ", "when ", "unless ", "except "))
    actor_clean = actor_clean and modal.group(1).split()[0] not in ("On", "Upon", "After", "Before", "By", "For", "With", "Without", "In", "At", "Under", "During", "Through", "Instead", "Nothing")
    if actor_clean:
        actor, verb, action = modal.groups()
        actor = actor[0].lower() + actor[1:] if actor.startswith(("The ", "An ", "Each ", "A ")) else actor
        focus = " ".join(action.split()[:9]).rstrip(".,;:")
        if (re.search(r"\b(that|which|who|whenever|when|if)\b|[—:]|\.\s+[A-Z]", actor, re.I)
                or re.match(r"(?:such|these|this|those|it\b|the provisions of this|any other information)", actor, re.I)
                or verb == "may" and action.startswith("be ")):
            return f"Under {unit['citation']}, what does the rule state concerning '{focus}'?", "source_scoped_rule"
        if "not" in verb:
            return f"In the {scope}, what restriction applies to {actor} concerning '{focus}'?", "restriction"
        if verb == "may":
            agent = re.search(r"\b(agency|official|employee|person|recipient|requester|applicant|director|department|commission|secretary|board|administrator|officer|state|you|licensee|entity|institution|organization|company|corporation|government|inspector|respondent|petitioner|individual|manufacturer|producer|operator|owner|dealer|carrier|contractor|committee|council|manager|authority|tribe|borrower|investigator|counsel|court|judge)\b", actor, re.I)
            if not agent:
                return f"In the {scope}, what rule applies to {actor} concerning '{focus}'?", "descriptive_rule"
            return f"In the {scope}, what is {actor} permitted to do concerning '{focus}', and subject to what conditions?", "permission"
        return f"In the {scope}, what requirement applies to {actor} concerning '{focus}'?", "obligation"
    timing = re.search(r"\b(within|no later than|at least|not more than|before|after)\b[^,;.]{0,80}?\b(days?|hours?|months?|years?)\b", value, re.I)
    if timing:
        phrase = normalized(timing.group(0))
        return f"For {unit['topic']} in the {scope}, what time requirement is stated as '{phrase}'?", "timing"
    if value.lower().startswith(("if ", "when ", "unless ", "until ", "except ", "notwithstanding ", "however")):
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
        if identifier not in mapping:
            return
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
        if identifier not in mapping:
            continue
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
    identifier = "title-2:section:180.605"
    add(identifier, "In the federal nonprocurement debarment and suspension rules, is suspension temporary or a final debarment determination?",
        "Suspension is a temporary status of ineligibility pending completion of an investigation or legal or debarment proceeding. Debarment is imposed for a specified period as a final determination that the person is not presently responsible.",
        ["Imposes suspension as a temporary status", "Imposes debarment for a specified period"], "reviewed_comparison")
    add(identifier, "What evidence and urgency findings distinguish suspension from debarment under the federal nonprocurement rules?",
        "For suspension, the official must have adequate evidence of a possible cause for debarment and conclude that immediate action is necessary to protect the Federal interest. Debarment requires a conclusion based on a preponderance of the evidence that the person engaged in conduct warranting debarment.",
        ["adequate evidence", "immediate action", "preponderance of the evidence"], "reviewed_comparison")
    add(identifier, "How does the timing of notice differ between federal nonprocurement suspension and debarment?",
        "Suspension is usually imposed first, followed by prompt notice and an opportunity to contest it. Debarment is imposed after the respondent receives notice and an opportunity to contest the proposed action.",
        ["Usually imposes the suspension", "Imposes debarment after"], "reviewed_comparison")
    identifier = "title-1:section:18.6"
    add(identifier, "What wording is used to certify a true copy of a document submitted for Federal Register filing and publication?",
        "Certified to be a true copy of the original.", ["Certified to be a true copy of the original"])
    add(identifier, "Who must sign the certification of a document copy submitted for Federal Register filing and publication?",
        "A certifying officer designated under section 16.1 of the chapter.", ["must be signed by a certifying officer designated under § 16.1"])
    add(identifier, "Are Presidential documents and duplicate originals subject to the copy-certification requirement stated in the Federal Register form-of-certification rule?",
        "No. The stated certification requirement excepts a Presidential document or a duplicate original.", ["except a Presidential document or a duplicate original"], "reviewed_exception")
    identifier = "title-1:section:21.45"
    add(identifier, "In Federal Register authority citations, does a nonstatutory authority citation come before or after statutory citations?",
        "After the statutory citations.", ["shall be placed after the statutory citations"])
    identifier = "title-1:section:21.52"
    add(identifier, "Must a Federal Register citation to statutory authority include a United States Code citation when one is available?",
        "Yes. All citations to statutory authority must include a United States Code citation where available.", ["shall include a United States Code citation, where available"])
    add(identifier, "If a United States Code citation is supplied as authority in a Federal Register document, are Public Law and U.S. Statutes at Large citations mandatory as well?",
        "No. The regulation makes Public Law and U.S. Statutes at Large citations optional when the United States Code is cited.", ["are optional when the United States Code is cited"], "reviewed_case")
    identifier = "title-1:section:21.53"
    add(identifier, "What information belongs in a citation to a nonstatutory document used in a Federal Register authority citation?",
        "The document designation and Federal Register volume and page, followed, if possible, by a parallel citation to the Code of Federal Regulations.", ["cited by document designation", "followed, if possible"])
    identifier = "title-1:section:22.6"
    add(identifier, "Where must the CFR citation identifying the affected area of a proposed regulatory action appear?",
        "Immediately below the name of the issuing agency.", ["immediately below the name of the issuing agency"])
    for identifier, agency in (("title-2:section:182.220", "the federal awarding agency"),
                                ("title-2:section:1401.320", "the Department of the Interior")):
        add(identifier, f"A new award recipient of {agency} has no existing drug-free workplace statement or awareness program, and the award performance period is less than 30 days. By when must they be in place?",
            "As soon as possible, but before the date on which performance is expected to be completed.",
            ["less than 30 days", "before the date on which performance is expected to be completed"], "reviewed_case")
        add(identifier, f"A new award recipient of {agency} has no existing drug-free workplace statement or awareness program, and the performance period is 30 days or more. What is the deadline?",
            "Within 30 days after award.", ["30 days or more", "within 30 days after award"], "reviewed_case")
        add(identifier, f"Can a new award recipient of {agency}, with no existing drug-free workplace statement or awareness program, obtain more than 30 days to establish them in extraordinary circumstances?",
            "The recipient may ask the awarding official for more time. Whether to grant additional time, and how much, is at the awarding official's discretion.",
            ["extraordinary circumstances", "more time to do so", "at the discretion"], "reviewed_case")
    identifier = "title-2:section:1800.10"
    add(identifier, "What does NASA stand for in the financial-assistance acronyms used by that agency?",
        "National Aeronautics and Space Administration.", ["NASA National Aeronautics and Space Administration"])
    add(identifier, "What does NFS stand for in NASA's financial-assistance acronyms?",
        "NASA FAR Supplement.", ["NFS NASA FAR Supplement"])
    identifier = "title-2:section:3001.400"
    add(identifier, "How does a National Science Foundation awarding official obtain a recipient's agreement to comply with the drug-free workplace guidance?",
        "By including the stated drug-free workplace term or condition in the award.", ["you must include the following term or condition in the award"])
    add(identifier, "Which drug-free workplace subpart is specified in the National Science Foundation award term when the recipient is an individual?",
        "Subpart C of 2 CFR Part 3001; the term specifies Subpart B for other recipients.", ["Subpart B (or Subpart C, if the recipient is an individual)"], "reviewed_case")
    identifier = "title-3:section:100.1"
    add(identifier, "Which ethical-conduct standards apply to employees of the Executive Office of the President?",
        "The executive branch-wide standards of ethical conduct at 5 CFR Part 2635.",
        ["standards of ethical conduct at 5 CFR part 2635"])
    add(identifier, "Which financial-disclosure regulations apply to employees of the Executive Office of the President?",
        "The executive branch-wide financial disclosure regulations at 5 CFR Part 2634.",
        ["financial disclosure regulations at 5 CFR part 2634"])
    add(identifier, "I am an Executive Office of the President employee. Which CFR parts govern my ethical conduct and financial disclosure?",
        "Ethical conduct is governed by 5 CFR Part 2635; financial disclosure is governed by 5 CFR Part 2634.",
        ["5 CFR part 2635", "5 CFR part 2634"], "reviewed_compound")
    identifier = "title-3:section:101.2"
    add(identifier, "Where does Title 3 direct a reader to find Office of Management and Budget Freedom of Information regulations?",
        "5 CFR Chapter III.", ["5 CFR Ch. III"])
    identifier = "title-3:section:102.102"
    add(identifier, "Do the Executive Office of the President's disability-nondiscrimination rules cover every agency-conducted program outside the United States?",
        "No. The regulation excepts programs or activities conducted outside the United States that do not involve individuals with disabilities in the United States.",
        ["except for programs or activities conducted outside the United States"], "reviewed_exception")
    identifier = "title-3:section:102.149"
    add(identifier, "Can inaccessible facilities justify excluding a qualified person with a disability from an Executive Office of the President agency program?",
        "The regulation prohibits exclusion, denial of benefits, or other discrimination on that basis, except as otherwise provided in section 102.150.",
        ["Except as otherwise provided in § 102.150", "no qualified individual with handicaps"], "reviewed_case")
    identifier = "title-3:section:102.151"
    add(identifier, "What accessibility requirement applies when an Executive Office of the President agency builds or alters a building for its use?",
        "The building or building part must be designed, constructed, or altered to be readily accessible to and usable by individuals with disabilities. The provision applies the Architectural Barriers Act, 42 U.S.C. 4151–4157, and the stated 41 CFR standards.",
        ["readily accessible to and usable", "Architectural Barriers Act"], "reviewed_case")
    return rows


def build(manifest, tokenizer, config):
    units = collect_units(manifest)
    unit_by_id = {unit["id"]: unit for unit in units}
    accepted, deferred, seen, inventory = [], [], set(), []
    limit = config["training"]["max_seq_length"]
    template_kwargs = config["model"]["chat_template_kwargs"]

    def emit(unit, question, answer, evidence, kind, method):
        question = normalized(question)
        if seen_key(question) in seen:
            return
        seen.add(seen_key(question))
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
        previous_rows = load_examples(original)
        initial_rows = [record for record in previous_rows if record.get("metadata", {}).get("question_type") == "initial_reviewed_fact"]
        if not initial_rows:
            initial_rows = previous_rows[:25]
        for record in initial_rows:
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
            for table in unit["tables"]:
                if table["complex"] or len(table["headers"]) < 2:
                    continue
                headers = [unit["nodes"][i]["text"] for i in table["headers"]]
                previous_values = {}
                for row_index, indices in enumerate(table["rows"]):
                    if len(indices) != len(headers):
                        continue
                    values = [clean(unit["nodes"][i]["text"]) for i in indices]
                    ditto_evidence = []
                    unresolved = False
                    for column, value in enumerate(values):
                        if re.fullmatch(r"(?:do\.?|ditto|same as above|\")", value, flags=re.I):
                            if column not in previous_values:
                                unresolved = True
                                break
                            values[column], source_index = previous_values[column]
                            ditto_evidence.append(source_index)
                        elif value:
                            previous_values[column] = (value, indices[column])
                    if unresolved:
                        continue
                    if not values or not values[0]:
                        continue
                    lead = [node["index"] for node in unit["nodes"] if node["tag"] == "P" and "table" in node["text"].lower()][:1]
                    lead_text = ""
                    if lead:
                        lead_text = unit["nodes"][lead[0]]["text"].strip()
                        lead_text = lead_text if lead_text.endswith(".") else lead_text.rstrip(":") + "."
                    evidence = sorted(set(table["headers"] + [i for i in indices if unit["nodes"][i]["text"]] + lead + ditto_evidence))
                    for column in range(1, len(headers)):
                        if not values[column] or values[column] in ("...", "*", "—", "–"):
                            continue
                        value_text = values[column]
                        column_evidence = list(evidence)
                        if CUT_END.search(value_text.strip()):
                            # Category cells introduce following sub-rows; include them
                            # instead of teaching a dangling header as the answer.
                            sub = []
                            for sub_row in table["rows"][row_index + 1:]:
                                cells = [unit["nodes"][i]["text"] for i in sub_row]
                                if cells and cells[0].strip():
                                    break
                                if len(cells) > 1 and cells[-1].strip():
                                    sub.append((sub_row[-1], cells[-1]))
                                if len(sub) >= 8:
                                    break
                            if not sub:
                                continue
                            column_evidence = sorted(set(column_evidence + [i for i, _ in sub]))
                            value_text = value_text.rstrip() + "\n" + "\n".join(
                                clean(t) for _, t in sub)
                            if CUT_END.search(value_text.strip()):
                                continue
                        if len(headers) == 2 and headers[0].strip().lower().startswith(("if", "when")):
                            condition = re.sub(r"^if\s+", "", values[0], flags=re.I)
                            question = f"{lead_text} In the {unit['topic']} table for the {unit['scope']}, what requirement applies if {condition[0].lower() + condition[1:]}?" if lead_text else f"In the {unit['topic']} table for the {unit['scope']}, what requirement applies if {condition[0].lower() + condition[1:]}?"
                        else:
                            key = "; ".join(f"{headers[i]}: {values[i]}" for i in range(len(headers)) if i != column and values[i])
                            question = f"{lead_text} In the {unit['topic']} table for the {unit['scope']}, what entry under '{headers[column]}' corresponds to the row '{key}'?" if lead_text else f"In the {unit['topic']} table for the {unit['scope']}, what entry under '{headers[column]}' corresponds to the row '{key}'?"
                        if len(values[column]) >= 25 and normalized(values[column]).lower() in normalized(question).lower():
                            continue
                        emit(unit, question, value_text, column_evidence, "table_lookup", "extractive")
            for node in unit["nodes"]:
                if node["tag"] in ("TH", "TD"):
                    continue
                if re.fullmatch(r"See\s+§+\s*[\d.]+\.?", clean(node["text"]), flags=re.I):
                    emit(unit, f"Which provision does the {unit['topic']} rule in the {unit['scope']} refer the reader to?",
                         clean(node["text"]), [node["index"]], "cross_reference", "extractive")
                    continue
                if len(clean(node["text"]).split()) < 4:
                    continue
                has_children = any(node["index"] in other["ancestors"] for other in unit["nodes"])
                if node["text"].endswith((":", "—", "–")) and not has_children:
                    # A complete preceding statement must not be discarded solely
                    # because its final sentence introduces an example or table.
                    if len(sentence_spans(node["text"])) < 2:
                        continue
                spans = sentence_spans(node["text"]) if not has_children else [(0, len(node["text"]), node["text"])]
                for sentence_index, (_, _, sentence) in enumerate(spans):
                    if sentence.endswith((":", "—", "–")) and not has_children:
                        continue
                    if len(clean(sentence).split()) < 5:
                        continue
                    question, kind = make_question(unit, node, sentence)
                    if not question:
                        continue
                    evidence = select_evidence(unit, node, descendants=has_children)
                    completed = extend_cut_evidence(
                        unit, [item["index"] for item in evidence])
                    if completed is None:
                        continue
                    evidence = [unit["nodes"][i] for i in completed]
                    # Keep complete paragraph/child context for the target, not a clipped sentence.
                    answer = "\n".join(clean(item["text"]) for item in evidence)
                    if len(spans) > 1:
                        # Focused answers use a complete source sentence plus any required lead-ins.
                        prefix = [item for item in evidence if item["index"] != node["index"]]
                        preserve_conditions = bool(re.search(r"\b(if|unless|except|however|notwithstanding|provided that|subject to|only if)\b", node["text"], re.I))
                        if preserve_conditions or sentence_index and sentence.lower().startswith(("however", "but ", "this ", "these ", "it ")):
                            answer = "\n".join(clean(item["text"]) for item in prefix + [node])
                        else:
                            answer = "\n".join([clean(item["text"]) for item in prefix] + [clean(sentence)])
                    if seen_key(question) in seen:
                        stem = question.rstrip("?")
                        question = f"{stem} (specifically {unit['citation']}{node['label']}, statement {sentence_index + 1})?"
                    emit(unit, question, answer, [item["index"] for item in evidence], kind, "extractive")
        inventory.append({"id": unit["id"], "part": unit["part_key"], "reserved": unit["reserved"],
                          "paragraphs": sum(node["tag"] == "P" for node in unit["nodes"]),
                          "text_blocks": len(unit["nodes"]), "tables": len(unit["tables"]),
                          "complex_tables_requiring_review": sum(table["complex"] for table in unit["tables"]),
                          "images_requiring_review": unit["images"], "new_usable_examples": len(accepted) - start,
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


def write_coverage_review(inventory, destination=WORK):
    review_queue = {
        "zero_auto_qa_units": [unit for unit in inventory if not unit["reserved"] and unit["text_blocks"] and not unit["new_usable_examples"]],
        "units_with_images": [unit for unit in inventory if unit["images_requiring_review"]],
        "units_with_complex_tables": [unit for unit in inventory if unit["complex_tables_requiring_review"]],
        "overlength_file": "overlength_review.jsonl",
        "note": "These entries are not certified complete by the text-only extractor. No content is invented to fill them.",
    }
    (Path(destination) / "coverage_review.json").write_text(json.dumps(review_queue, indent=2), encoding="utf-8")
    return review_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--reuse-sources", action="store_true")
    parser.add_argument("--titles", type=parse_titles, default=[1, 2],
                        help="CFR title range, e.g. 1-10 or 1,2,3-7.")
    args = parser.parse_args()
    config = load_config(ROOT)
    paths = runtime_paths(config, ROOT)
    source_manifest = WORK / "source_manifest.json"
    manifest = json.loads(source_manifest.read_text()) if args.reuse_sources and source_manifest.exists() else None
    present = {source["title"] for source in manifest["sources"]} | set(manifest.get("reserved_titles", [])) if manifest else set()
    if manifest is None or not set(args.titles) <= present:
        manifest = fetch_sources(args.titles)
    if set(args.titles) != {source["title"] for source in manifest["sources"]} | set(manifest.get("reserved_titles", [])):
        manifest = {**manifest, "sources": [source for source in manifest["sources"] if source["title"] in args.titles]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths["model"], local_files_only=True)
    records, deferred, inventory = build(manifest, tokenizer, config)
    old_manifest = json.loads((ROOT / "data" / "split_manifest.json").read_text())
    splits, assignments, removed = assign_splits(records, old_manifest)

    def write_rows(file, rows):
        temporary = file.with_name(file.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(file)

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
        "qa_by_title": dict(Counter(record["metadata"]["source_id"].split(":")[0] for record in records)),
        "requested_titles": args.titles,
        "images_requiring_review": sum(unit["images_requiring_review"] for unit in inventory),
        "complex_tables_requiring_review": sum(unit["complex_tables_requiring_review"] for unit in inventory),
        "review_limit": "Evidence is verified for every example. Templates are extractive, not a claim of independent legal interpretation. Overlength complete contexts are deferred, never truncated.",
        "training_started": False,
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_coverage_review(inventory)
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
        "version": "detailed_ecfr_v3", "titles": args.titles, "seed": 3407, "total_examples": sum(map(len, splits.values())),
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
