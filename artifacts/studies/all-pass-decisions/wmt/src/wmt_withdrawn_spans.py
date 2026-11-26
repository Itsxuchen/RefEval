"""Locate Sonnet and first-pass human major-error spans on withdrawn WMT23 approvals.

This exploratory comparison uses the corrected sealed frame. Its unit for percentages
is an auditor / output / realized priority order in which the early verdict was
withdrawn. Matching quoted positions is evidence about marks, not an annotation
of whether their underlying linguistic explanations agree.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re

import numpy as np

from src import wmt_blind_crossover as study
from src import wmt_blind_inputs as source


@dataclass(frozen=True)
class Mark:
    axis: int
    side: str
    span: tuple[int, int] | None


def _trim(text, left, right):
    while left < right and text[left].isspace():
        left += 1
    while left < right and text[right - 1].isspace():
        right -= 1
    return (left, right) if left < right else None


def human_mark(axis, src, tgt):
    found = []
    for side, raw in (("source", src), ("target", tgt)):
        normalized = raw.replace("\u200b", "")
        pieces = re.split(r"(</?v>)", normalized)
        unmarked = "".join(x for x in pieces if x not in ("<v>", "</v>"))
        left_trim = len(unmarked) - len(unmarked.lstrip())
        opening = None
        pos = 0
        for piece in pieces:
            if piece == "<v>":
                if opening is not None:
                    raise ValueError("Nested human span")
                opening = pos
            elif piece == "</v>":
                if opening is None:
                    raise ValueError("Unpaired human span")
                interval = _trim(unmarked, opening, pos)
                coords = tuple(i-left_trim for i in interval) if interval else None
                found.append(Mark(axis, side, coords))
                opening = None
            else:
                pos += len(piece)
        if opening is not None or unmarked.strip() != source._txt(raw):
            raise ValueError("Human span text mismatch")
    return found or [Mark(axis, "unknown", None)]


def match_locations(text, phrase):
    return [m.start() for m in re.finditer("(?="+re.escape(phrase)+")", text)] if phrase else []


def machine_mark(error, pair):
    axis = source._fail_axis(error["category"], error["severity"])
    if axis is None:
        return None
    side = "source" if error["is_source_error"] else "target"
    text = pair[side == "target"]
    phrase = source._txt(error["span"] or "")
    hits = match_locations(text, phrase)
    if len(hits) > 1:
        context = source._txt(error["extended_span"] or "")
        intervals = [(i, i+len(context)) for i in match_locations(text, context)]
        contextual = [i for i in hits if any(lo <= i and i+len(phrase) <= hi for lo, hi in intervals)]
        if len(contextual) == 1:
            hits = contextual
    span = (hits[0], hits[0]+len(phrase)) if len(hits) == 1 else None
    return Mark(axis, side, span)


def records(cache, targets):
    """Load only relevant positive marks, respecting the immutable unit/text join."""
    humans = defaultdict(list)
    texts = {}
    for line in source._source_rows(cache / "ende_hum.tgz"):
        key = source._keys(line)
        if key not in targets:
            continue
        fields = line.decode().split("\t", 9)
        unit = (key, fields[4])
        if fields[4] not in targets[key]:
            continue
        pair = (source._txt(fields[5]), source._txt(fields[6]))
        if key in texts and texts[key] != pair:
            raise ValueError("Raters disagree on original text identity")
        texts[key] = pair
        axis = source._fail_axis(fields[7], fields[8])
        if axis is not None:
            humans[unit].extend(human_mark(axis, fields[5], fields[6]))

    wanted = set(texts.values())
    predictions = {}
    with gzip.open(cache / "sonnet.gz", "rt", encoding="utf-8") as f:
        for line in f:
            machine, sep, _ = line.partition(', "human_evaluation":')
            if not sep:
                raise ValueError("Unexpected serialization of human/automatic scores")
            row = json.loads(machine+"}")
            if (row["src_lang"], row["tgt_lang"]) != ("English","German"):
                continue
            pair = (source._txt(row["src"]), source._txt(row["tgt"]))
            if pair not in wanted:
                continue
            marks = tuple(m for err in row["evaluation"]["errors"] if (m:=machine_mark(err,pair)) is not None)
            if pair in predictions and set(predictions[pair]) != set(marks):
                raise ValueError("Repeated machine text with unmatched spans")
            predictions[pair] = marks
    if set(predictions) != wanted:
        raise ValueError("Missing machine marks")
    return humans, texts, predictions


def classify(pair, expert, machine):
    best = 0.0
    token_overlap = False
    comparison = False
    for a in expert:
        for j in machine:
            if a.side != j.side or a.span is None or j.span is None:
                continue
            comparison = True
            text = pair[a.side == "target"]
            al, ar = a.span
            jl, jr = j.span
            assert 0 <= al < ar <= len(text) and 0 <= jl < jr <= len(text)
            # A word touched by both marks can match even for disjoint character ranges.
            token_overlap |= any(w.start()<ar and w.end()>al and w.start()<jr and w.end()>jl
                                 for w in re.finditer(r"\w+",text))
            cross = sum(not c.isspace() for c in text[max(al,jl):min(ar,jr)])
            if cross:
                union = sum(not c.isspace() for c in text[min(al,jl):max(ar,jr)])
                best = max(best, cross/union)
    located = all(x.span is not None for x in (*expert,*machine))
    comparable_sides = bool({x.side for x in expert}&{x.side for x in machine})
    strict = any(a.side==j.side and a.span is not None and a.span==j.span for a in expert for j in machine)
    return dict(exact_span=strict, iou50=best>=0.5, iou25=best>=0.25, any_nonspace_char=best>0,
                any_word_touched=token_overlap, completely_located=located,
                has_located_same_side_pair=comparison,
                only_different_sides=not comparable_sides and located,
                no_overlap_all_located_comparable=(best==0 and located and comparable_sides),
                unresolved_no_observed_overlap=(best==0 and not located))


def analyze(cache):
    held = study.read(study.OUT / "seal.json")["split"]["held"]
    rows, j, h, masks = study.frame(cache,held)
    eager, gated = study.policies(j,h,masks)
    early = eager & ~gated
    withdrawn = early & ~h.all(axis=2)
    counts = withdrawn.sum(axis=0)
    id_map = source._frame(cache)[0]
    targets = defaultdict(dict)
    for i, a in zip(*np.nonzero(counts)):
        doc, segment, system = rows[i].sid.split("|")
        key = system, doc, segment
        rater = id_map[key][2][a]
        targets[key][rater] = (int(counts[i,a]), set(np.flatnonzero(~j[i])), set(np.flatnonzero(~h[i,a])))

    human, texts, predictions = records(cache,targets)
    docs = sorted(held)
    strata = {doc:Counter() for doc in docs}
    rater_counts = Counter()
    axis_sets = Counter()
    unit_sets = defaultdict(set)
    for key, ratings in targets.items():
        pair = texts[key]
        automatic = predictions[pair]
        for rater, (num, jaxes, haxes) in ratings.items():
            expert = human[(key,rater)]
            assert {m.axis for m in expert} == haxes and {m.axis for m in automatic} == jaxes
            assert haxes and jaxes and not (haxes & jaxes)
            flags = classify(pair, expert, automatic)
            names = [k for k,v in flags.items() if v] + ["different_nonempty_axis_sets","total"]
            rater_counts.update(names)
            axis_sets[",".join(source.AXES[i] for i in sorted(jaxes)) + " -> " +
                      ",".join(source.AXES[i] for i in sorted(haxes))] += num
            for n in names:
                unit_sets[n].add(key)
                strata[key[1]][n] += num
    flags = list(classify(("",""),[],[])) + ["different_nonempty_axis_sets","total"]
    totals = sum(strata.values(), Counter())
    assert totals["total"] == int(withdrawn.sum()) and totals["total"] == sum(axis_sets.values())
    assert sum(totals[k] for k in ("any_nonspace_char","only_different_sides",
                                   "no_overlap_all_located_comparable","unresolved_no_observed_overlap")) == totals["total"]
    assert totals["exact_span"]<=totals["iou50"]<=totals["iou25"]<=totals["any_nonspace_char"]
    baseline = study.read(study.OUT/"held.json")["estimates"]["withdrawn_per100"]["point"]
    np.testing.assert_allclose(100*totals["total"]/(3*31*len(rows)), baseline)
    draws=np.random.default_rng(20261125).multinomial(len(docs), np.full(len(docs),1/len(docs)), size=1999)
    denominator=draws @ np.array([strata[d]["total"] for d in docs])
    defined=denominator > 0
    summary={}
    for n in flags:
        samples=(draws @ np.array([strata[d][n] for d in docs]))[defined]*100/denominator[defined]
        summary[n]=dict(orders=totals[n], auditor_outputs=rater_counts[n], distinct_outputs=len(unit_sets[n]),
                        percent_orders=100*totals[n]/totals["total"],
                        pct_interval=np.percentile(samples,[2.5,97.5]).tolist(),
                        defined_draws=int(defined.sum()))
    return dict(scope="exploratory; conditional on withdrawals under corrected sealed masks",
                docs=len(docs),outputs=len(rows),orders_per_auditor=31,auditors_per_output=3,
                early_auditor_orders=int(early.sum()), withdrawn_auditor_orders=int(withdrawn.sum()),
                withdrawn_output_equivalents=totals["total"]/93,findings=summary,
                axis_set_pair_orders=dict(sorted(axis_sets.items())),
                method="Cross-side marks are not aligned. IoU compares non-whitespace character positions on the same original text. A positive classification requires at least one qualifying cross-axis pair; it is not a semantic adjudication.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--out",type=Path,default=study.ROOT/"artifacts/reports/wmt_followup_20261126/spans.json")
    args=p.parse_args()
    output=analyze(args.cache)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    study.store(args.out, output)
    print(json.dumps(output,indent=2))
