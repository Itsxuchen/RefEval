"""Build text-free, article-separated SummEval judge panels from pinned releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reports/external_frame_20261125"
DEFAULT_CACHE = Path("/private/tmp/refeval_summeval_20261125")
DIMS = ("coherence", "consistency", "fluency", "relevance")
MODELS = ("G-Eval GPT-4", "UniEval")
CUTOFF = np.array([[4., 4., 2.5, 4.], [.5, .5, .5, .5]])
GE_BASE = "https://raw.githubusercontent.com/nlpyang/geval/6f84404ce4503d2ba0cde0293c5d402ecf2976a8/results"
UNI_BASE = "https://raw.githubusercontent.com/maszhongming/UniEval/d33e7b6cfebe97b2bafe435adbd818230d5a416a/reproduce"
SOURCES = {
    "experts.jsonl": ("https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl",
                      "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab"),
    "summeval_result.json": (UNI_BASE + "/unieval_predict/summarization/summeval_result.json",
                             "71f437a9253191fc1b6fa558fbeca432896e2c8c16d46b460593e0546945075b"),
}
for prefix, digest in zip(("coh", "con", "flu", "rel"), (
        "bec62a6a8408b7f0c7a55a665d8c9a44b97eecf671bf0b4319284b192d218508",
        "e09609ab249fc6aa2e4f11b7c16d7531b7036b163c8251e54a04ce4c3efa76ed",
        "4f08454a9ac1c3c58f6e4d067a00ed37d83d4de2206b010d8a1ba6ef18ceae7d",
        "c222933ff5ba552bb9ec3d2b495022a89b8019cf246710988a4e49804d47585a")):
    name = f"gpt4_{prefix}_detailed.json"
    SOURCES[name] = (GE_BASE + "/" + name, digest)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sources(cache):
    cache.mkdir(parents=True, exist_ok=True)
    indexed = {}
    for name, (url, digest) in SOURCES.items():
        path = cache / name
        if not path.exists():
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            if hashlib.sha256(r.content).hexdigest() != digest:
                raise ValueError("Remote source hash mismatch: " + name)
            path.write_bytes(r.content)
        if sha(path) != digest:
            raise ValueError("Cached source hash mismatch: " + name)
        raw = path.read_text()
        rows = [json.loads(l) for l in raw.splitlines()] if name.endswith(".jsonl") else json.loads(raw)
        ids = ("id", "model_id") if name.endswith(".jsonl") else ("doc_id", "system_id")
        indexed[name] = {(r[ids[0]], r[ids[1]]):r for r in rows}
        if len(indexed[name]) != len(rows) or len(rows) != 1600:
            raise ValueError("Incomplete or duplicated released identities")
    if any(set(v) != set(indexed["experts.jsonl"]) for v in indexed.values()):
        raise ValueError("Incompatible summary keys")
    for k, record in indexed["experts.jsonl"].items():
        original = canonical_text(record["decoded"])
        uni = indexed["summeval_result.json"][k]
        if original != canonical_text(uni["system_output"]):
            raise ValueError("SummEval expert and UniEval text content differs")
        rendered = {f: unicode_as_windows1252(uni[f]) for f in ("source", "reference", "system_output")}
        for name, data in indexed.items():
            if name.startswith("gpt4_") and any(
                    data[k][f] != rendered[f] for f in rendered):
                raise ValueError("Different text between judges")
    return indexed


def canonical_text(s):
    s = s.casefold()
    for token in ("-lrb-", "-rrb-", "-lsb-", "-rsb-", "-lcb-", "-rcb-"):
        s = s.replace(token, "")
    return "".join(ch for ch in unicodedata.normalize("NFKC", s) if ch.isalnum())


# The G-Eval result JSON decodes UTF-8 bytes with the Windows-1252 display
# mapping. Undefined Windows-1252 bytes remain the same control codepoints.
_DISPLAY_BYTES = tuple(bytes([b]).decode("cp1252") if b not in (129,141,143,144,157)
                       else chr(b) for b in range(256))


def unicode_as_windows1252(s):
    return "".join(_DISPLAY_BYTES[b] for b in s.encode("utf-8"))


def partition(ids):
    def priority(doc_id):
        return hashlib.sha256(("summeval-selection-v1|" + doc_id).encode()).hexdigest(), doc_id
    docs = sorted({d for d, _ in ids}, key=priority)
    if len(docs) != 100:
        raise ValueError("Unexpected source article count")
    return set(docs[:30]), set(docs[30:])


def reply_mean(replies, maximum):
    if len(replies) != 20:
        raise ValueError("Incomplete G-Eval response list")
    scores = []
    for r in replies:
        match = re.search(r"^ ?([\d\.]+)", r)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 1 <= value <= maximum:
            scores.append(value)
    return (np.mean(scores) if len(scores) >= 16 else np.nan), 20-len(scores)


def decode_grouped_subset(sources, groups):
    kept, missing, rejected = [], Counter(), np.zeros(4, dtype=int)
    for key, e in sorted(sources["experts.jsonl"].items()):
        if key[0] not in groups:
            continue
        ratings = np.array([[r[d] for d in DIMS] for r in e["expert_annotations"]])
        if ratings.shape != (3,4) or not np.isin(ratings, np.arange(1,6)).all():
            missing["expert"] += 1
            continue
        med = np.median(ratings, axis=0)
        ge = []
        for n,(abbr,top) in enumerate(zip(("coh","con","flu","rel"), (5,5,3,5))):
            mean, bad = reply_mean(sources[f"gpt4_{abbr}_detailed.json"][key]["all_responses"], top)
            ge.append(mean)
            rejected[n] += bad
        uni = [sources["summeval_result.json"][key]["predict_scores"][d] for d in DIMS]
        values = np.array([ge, uni], dtype=float)
        if not np.isfinite(values).all() or any(x < 0 or x > 1 for x in uni):
            missing["judge"] += 1
            continue
        kept.append((key, med, values))
    return kept, dict(missing), rejected.tolist()


def calibrate(rows):
    scores = np.array([values for _,_,values in rows])
    gold = np.array([med >= 4 for _,med,_ in rows])
    result = CUTOFF.copy()
    for j in range(2):
        for d in range(4):
            s = scores[:,j,d]
            t = np.unique(np.r_[s, np.nextafter(s.max(), np.inf), CUTOFF[j,d]])
            n = ((s[:,None] >= t) == gold[:,d,None]).sum(axis=0)
            result[j,d] = min(t[n == n.max()], key=lambda x: (abs(x-CUTOFF[j,d]), -x))
    return result


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2, allow_nan=False) + "\n")


def run(stage, cache, seal=None):
    proto = OUT / "pre_outcome_protocol.md"
    freeze = OUT / "pre_outcome_freeze_v2.json"
    original = OUT / "pre_outcome_freeze.json"
    erratum = OUT / "encoding_preflight.md"
    if stage == "freeze":
        if freeze.exists():
            raise FileExistsError("Design already frozen")
        dump(freeze, dict(protocol_sha256=sha(proto), loader_sha256=sha(Path(__file__)),
                          supersedes_sha256=sha(original), metadata_amendment_sha256=sha(erratum),
                          sources=SOURCES, definition="Before decoding development or held expert/judge values"))
        return
    frozen = json.loads(freeze.read_text())
    if frozen["protocol_sha256"] != sha(proto) or frozen["loader_sha256"] != sha(Path(__file__)):
        raise ValueError("Frozen design or loader changed")
    if stage == "held" and (seal is None or not seal.is_file()):
        raise ValueError("Development-choice seal required for held extraction")
    sources = load_sources(cache)
    train, test = partition(sources["experts.jsonl"])
    if stage == "held":
        pre = json.loads((OUT / "development_manifest.json").read_text())
        dev_hash = sha(OUT / "development_bits.json")
        if pre["panel_sha256"] != dev_hash or dev_hash not in seal.read_text():
            raise ValueError("Choice seal must identify the exact development panel")
    rows, missing, rejected = decode_grouped_subset(sources, train if stage == "development" else test)
    cutoffs = calibrate(rows) if stage == "development" else np.array(pre["calibrated_cutoffs"])
    panels = {}
    for name, thresh in (("primary", CUTOFF), ("development_calibrated", cutoffs)):
        panels[name] = [dict(unit=key[0]+"/"+key[1], group=key[0], reference=(med>=4).astype(int).tolist(),
                             predictions={m:(v[i]>=thresh[i]).astype(int).tolist() for i,m in enumerate(MODELS)})
                        for key,med,v in rows]
    file = OUT / (stage + "_bits.json")
    dump(file, dict(models=MODELS, dimensions=DIMS, panels=panels))
    manifest = dict(panel_sha256=sha(file), design_sha256=sha(freeze), n_outputs=len(rows),
                    n_articles=len({key[0] for key,_,_ in rows}), excluded=missing,
                    invalid_geval_samples_by_dimension=rejected, primary_cutoffs=CUTOFF.tolist(),
                    calibrated_cutoffs=cutoffs.tolist())
    if stage == "held":
        manifest["choice_seal_sha256"] = sha(seal)
        meds = np.array([med for _,med,_ in rows])
        pred = np.array([v for _,_,v in rows])
        manifest["spearman_by_judge_dimension"] = {
            m:[float(spearmanr(pred[:,i,d], meds[:,d]).statistic) for d in range(4)]
            for i,m in enumerate(MODELS)}
    dump(OUT/(stage + "_manifest.json"), manifest)
    print(stage, len(rows), manifest["n_articles"], manifest["panel_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","development","held"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--seal", type=Path)
    args = parser.parse_args()
    run(args.stage, args.cache, args.seal)
