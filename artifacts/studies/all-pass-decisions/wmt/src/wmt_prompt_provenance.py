"""Verify recorded WMT judge requests against the pinned public prompt text."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from src import wmt_blind_inputs as source
from src.wmt_blind_crossover import OUT as CONFIRMATION

URL = ("https://raw.githubusercontent.com/amazon-science/span-mt-metaeval/"
       + source.MACHINE_COMMIT + "/src/mt_evaluation/autoevals/unified/unified_mqm_boosted_v5.py")
TEMPLATE_SHA256 = "ae9efd7d334c2507b757d6d093ba09e8f090822fe7db80593b29f50e2998f461"
OUT = CONFIRMATION.parents[1] / "wmt_followup_20261126"


def template():
    with urlopen(URL, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("Unexpected publisher template.")
    tree = ast.parse(data)
    prompt, = [x for x in ast.walk(tree) if isinstance(x, ast.Call)
               and isinstance(x.func, ast.Name) and x.func.id == "Prompt"]
    fields = {p.arg: ast.literal_eval(p.value) for p in prompt.keywords}
    function, = [x for x in ast.walk(tree)
                 if isinstance(x, ast.FunctionDef) and x.name == "get_final_instruction"]
    instructions = [ast.literal_eval(node.value) for node in ast.walk(function)
                    if isinstance(node, ast.Return)]
    return fields, instructions


def held_texts(cache):
    docs = set(json.loads((CONFIRMATION / "seal.json").read_text())["split"]["held"])
    frame, _ = source._frame(cache)
    texts = {}
    for line in source._source_rows(cache/"ende_hum.tgz"):
        key = source._keys(line)
        if key in frame and key[1] in docs:
            meta = line.split(b"\t", 7)
            texts[key] = (source._txt(meta[5].decode()), source._txt(meta[6].decode()))
    return texts


def check(cache):
    fields, instructions = template()
    assert fields["few_shots"] == []
    units = held_texts(cache)
    wanted = set(units.values())
    report = {}
    for name in ("sonnet", "haiku"):
        seen = set()
        matched = total = 0
        with gzip.open(cache/f"{name}.gz", "rt") as stream:
            for line in stream:
                prefix, sep, _ = line.partition(', "human_evaluation":')
                if not sep:
                    raise ValueError("Unexpected cache layout.")
                row = json.loads(prefix + "}")
                key = (source._txt(row["src"]), source._txt(row["tgt"]))
                if (row["src_lang"],row["tgt_lang"]) != ("English","German") or key not in wanted:
                    continue
                e = row["evaluation"]
                seen.add(key)
                total += 1
                content = {k:row[k] for k in ("src","tgt","src_lang","tgt_lang")}
                has_match = any(fields["user_prompt"].format(**content, final_instruction=instruction)
                                == e["user_prompt"] for instruction in instructions)
                if (e["system_prompt"] == fields["system_prompt"] and has_match
                    and e["few_shots"] == {"user_prompts":[],"assistant_responses":[]}):
                    matched += 1
        if seen != wanted or matched != total:
            raise ValueError(f"Archived {name} request failed reconstruction: {matched}/{total}")
        report[name] = dict(cache_rows=total, exact_system_user_zero_shot=matched,
                            distinct_input_pairs=len(seen), held_output_keys=len(units))
    return dict(commit=source.MACHINE_COMMIT, template_sha256=TEMPLATE_SHA256, records=report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=OUT/"prompts.json")
    args = parser.parse_args()
    result = check(args.cache)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(result)
