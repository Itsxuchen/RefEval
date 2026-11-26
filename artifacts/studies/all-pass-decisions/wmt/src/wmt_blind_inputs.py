"""Read the pinned WMT23 professional first-pass panel and two archived MQM judges.

inventory() joins identifier/text metadata and model parsing status. Human categories
and severities are decoded only by load_units(), for its caller's named documents.
The separate second-pass file supplies three identifier columns only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

HUMAN_COMMIT = "29acd6999aaea3586378c9dcb58d8cc4820cede3"
MACHINE_COMMIT = "ca494e2626fd85f440ba3934404d347b281bdfcc"
GOOGLE = f"https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/{HUMAN_COMMIT}/generalMT2023/ende"
AMAZON = f"https://raw.githubusercontent.com/amazon-science/span-mt-metaeval/{MACHINE_COMMIT}/outputs/unified-mqm-boosted-v5"
SOURCES = {
    "ende_hum.tgz": (f"{GOOGLE}/mqm_generalMT2023_ende.3ratingsPerSegment.tsv.tar.gz", "c4d453a237cf87ebc0e1e7fd6279d6508670376e"),
    "ende_human_frame.tsv": (f"{GOOGLE}/mqm_re_annotation_generalMT2023_ende.human_prior_rater.tsv", "90c392b2d5f245631f424a49ddcacf4bf49d866a"),
    "sonnet.gz": (f"{AMAZON}/global.anthropic.claude-sonnet-4-5-20250929-v1:0/wmt23/cache.jsonl.gz", "929ba86c9546e7640ae9f1a02bdbe61269e20b47"),
    "haiku.gz": (f"{AMAZON}/global.anthropic.claude-haiku-4-5-20251001-v1:0/wmt23/cache.jsonl.gz", "4e7b8f2963816a499955d3b37d40a7448321b3e8"),
}
AXES = ("accuracy", "fluency", "terminology", "other_target")
EXCLUDED_DOCS = frozenset({"news_guardian.114893", "news_aj-english.33941"})
FIRST_PASS = "mqm_generalMT2023_ende.3ratingsPerSegment.tsv"


@dataclass(frozen=True)
class Unit:
    doc: str
    sid: str
    primary: tuple[bool, ...]
    auxiliary: tuple[bool, ...]
    experts: tuple[tuple[bool, ...], ...]


def _blob_digest(path: Path) -> str:
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(cache: Path) -> None:
    """Fetch absent originals, checking the git blob even if a file is cached."""
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        dest = cache / name
        if dest.is_file():
            if _blob_digest(dest) != expected:
                raise ValueError(f"Original WMT source hash mismatch: {name}")
            continue
        with NamedTemporaryFile(dir=cache, delete=False) as stream:
            tmp = Path(stream.name)
        try:
            urllib.request.urlretrieve(url, tmp)
            if _blob_digest(tmp) != expected:
                raise ValueError(f"Downloaded WMT source hash mismatch: {name}")
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)


def _doc_name(doc: str) -> str:
    return re.sub(r"(?::en-de)+$", "", doc)


def _txt(s: str) -> str:
    # The rater UI inserts <v> for highlights and zero-width spaces at token
    # boundaries. Apply the same trimming and boundary normalization on both sides.
    return s.replace("<v>", "").replace("</v>", "").replace("\u200b", "").strip()


def _source_rows(path: Path):
    with tarfile.open(path, "r:gz") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        if names != [FIRST_PASS]:
            raise ValueError(f"Unexpected first-pass archive: {names}")
        with tar.extractfile(FIRST_PASS) as stream:
            header = stream.readline().split(b"\t")
            if header[:9] != b"system doc docSegId globalSegId rater source target category severity".split():
                raise ValueError("Unexpected Single TSV schema")
            yield from stream


def _keys(line: bytes) -> tuple[str, str, str]:
    system, doc, sid, _ = line.split(b"\t", 3)
    return system.decode(), _doc_name(doc.decode()), sid.decode()


def _fail_axis(category: str, severity: str) -> int | None:
    if severity.strip().lower() not in {"major", "critical"}:
        return None
    top = re.split(r"[/\-]", category.strip().lower(), maxsplit=1)[0].strip()
    folded = re.sub(r"[^a-z]", "", top)
    if folded in {"source", "sourceerror", "sourceissue", "no", "noerror", "neutral", "hotw", "hotwtest", "attentioncheck"}:
        return None
    if folded in AXES[:3]:
        return AXES.index(folded)
    return 3


def _pass_vector(pairs) -> tuple[bool, ...]:
    failed = {_fail_axis(category, severity) for category, severity in pairs}
    return tuple(i not in failed for i in range(len(AXES)))


def _machine(path: Path, wanted: set[tuple[str, str]]):
    found = {}
    ambiguous = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            # The cache contains a redundant embedded human_evaluation field.
            # Never interpret it. Only the archived evaluation is used.
            machine_json, sep, _ = line.partition(', "human_evaluation":')
            if not sep:
                raise ValueError("Unexpected machine cache serialization")
            row = json.loads(machine_json + "}")
            if (row["src_lang"], row["tgt_lang"]) != ("English", "German"):
                continue
            key = _txt(row["src"]), _txt(row["tgt"])
            if key not in wanted:
                continue
            record = row["evaluation"]
            if record["parsing_error"] or not isinstance(record["errors"], list):
                ambiguous.add(key)
                continue
            vector = _pass_vector((e["category"], e["severity"]) for e in record["errors"])
            if key in found and found[key] != vector:
                ambiguous.add(key)
            found[key] = vector
    return {key: bits for key, bits in found.items() if key not in ambiguous}


@lru_cache(maxsize=2)
def _frame(cache: Path):
    prepare(cache)
    with (cache / "ende_human_frame.tsv").open("rb") as stream:
        all_keys = {_keys(line) for line in stream if line.strip()}
    if any(not key[1].startswith("news_") for key in all_keys):
        raise ValueError("Unexpected non-news key in Riley frame")
    wanted = {k for k in all_keys if not k[0].lower().startswith("ref") and k[1] not in EXCLUDED_DOCS}
    rater_sets = defaultdict(set)
    human_text = defaultdict(set)
    for line in _source_rows(cache / "ende_hum.tgz"):
        key = _keys(line)
        if key not in wanted:
            continue
        fields = line.split(b"\t", 7)
        rater_sets[key].add(fields[4].decode())
        human_text[key].add((_txt(fields[5].decode()), _txt(fields[6].decode())))
    complete = {k: next(iter(human_text[k])) for k in wanted
                if len(rater_sets[k]) == 3 and len(human_text[k]) == 1}
    text_keys = set(complete.values())
    sonnet = _machine(cache / "sonnet.gz", text_keys)
    haiku = _machine(cache / "haiku.gz", text_keys)
    # Prevent exact duplicate input pairs from appearing in different document splits.
    text_docs = defaultdict(set)
    for key, text in complete.items():
        text_docs[text].add(key[1])
    frame = {k: (sonnet[t], haiku[t], tuple(sorted(rater_sets[k]))) for k, t in complete.items()
             if t in sonnet and t in haiku and len(text_docs[t]) == 1}
    stats = dict(all_riley_keys=len(all_keys), riley_documents=len({k[1] for k in all_keys}),
                 excluded_reference=len([k for k in all_keys if k[0].lower().startswith("ref")]),
                 nonref_in_excluded_doc=sum(k[1] in EXCLUDED_DOCS and not k[0].lower().startswith("ref") for k in all_keys),
                 wanted=len(wanted), human_three_raters=sum(len(rater_sets[k]) == 3 for k in wanted),
                 human_three_consistent_text=len(complete), sonnet_matching=sum(t in sonnet for t in complete.values()),
                 haiku_matching=sum(t in haiku for t in complete.values()),
                 crossed_documents=sum(len(text_docs[t]) > 1 for t in complete.values()),
                 eligible=len(frame))
    return frame, stats


def inventory(cache: Path) -> dict[str, int]:
    """Counts by source document, prior to accessing any first-pass ratings."""
    return dict(sorted(Counter(k[1] for k in _frame(Path(cache))[0]).items()))


def flow(cache: Path) -> dict[str, int]:
    return _frame(Path(cache))[1].copy()


def load_units(cache: Path, allowed_docs) -> list[Unit]:
    cache = Path(cache)
    frame, _ = _frame(cache)
    allowed = set(allowed_docs)
    if not allowed or not allowed.issubset({k[1] for k in frame}):
        raise ValueError("Supply a nonempty subset of eligible document IDs")
    selected = {k: v for k, v in frame.items() if k[1] in allowed}
    by_rater = defaultdict(list)
    for line in _source_rows(cache / "ende_hum.tgz"):
        key = _keys(line)
        if key not in selected:
            continue  # Never decode rating fields of a document outside this phase.
        fields = line.split(b"\t", 9)
        by_rater[(key, fields[4].decode())].append((fields[7].decode(), fields[8].decode()))
    result = []
    for (system, doc, local_id), (judge, aux, raters) in sorted(selected.items()):
        key = system, doc, local_id
        if any((key, rater) not in by_rater for rater in raters):
            raise ValueError("A selected Single rating vanished")
        refs = tuple(_pass_vector(by_rater[key, rater]) for rater in raters)
        result.append(Unit(doc, f"{doc}|{local_id}|{system}", judge, aux, refs))
    return result
