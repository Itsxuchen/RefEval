"""Post-hoc peer-role heterogeneity and human-prior second-pass scoring on the sealed WMT holdout."""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src import wmt_blind_crossover as study
from src import wmt_blind_inputs as inputs


def read_second(cache, keys):
    """Read final annotations only; deleted/prior flags never become final errors."""
    second = defaultdict(list)
    priors = defaultdict(set)
    docs = set()
    for line in (cache / "ende_human_frame.tsv").open("rb"):
        unit_key = inputs._keys(line)
        docs.add(unit_key[1])
        if unit_key not in keys:
            continue
        fields = line.decode().rstrip("\n").split("\t")
        identity = (*unit_key, fields[4])
        meta = json.loads(fields[9])
        priors[identity].add(meta["prior_rater"])
        second[identity].append((fields[7], fields[8]))
    if any(len(s) != 1 for s in priors.values()):
        raise ValueError("Conflicting prior rater identity in the same second-pass rating")
    return {k: (next(iter(priors[k])), all(inputs._pass_vector(v))) for k, v in second.items()}, docs


def check_prior_spans(cache):
    """Check the artificial-span exclusion against both the frame and the archived prior traces."""
    originals = defaultdict(set)
    first_docs = set()
    for line in inputs._source_rows(cache / "ende_hum.tgz"):
        key = inputs._keys(line)
        if not key[1].startswith("news_"):
            continue
        first_docs.add(key[1])
        fields = line.decode().rstrip("\n").split("\t")
        md = json.loads(fields[9])
        if fields[7].lower().startswith(("hotw", "no-error", "no error")):
            continue
        for side in ("source_spans", "target_spans"):
            for lo, hi in md.get(side, []):
                originals[key, fields[4], side].update(range(lo, hi+1))
    second_docs = set()
    traces = 0
    unmatched = Counter()
    for line in (cache / "ende_human_frame.tsv").open("rb"):
        key = inputs._keys(line)
        second_docs.add(key[1])
        md = json.loads(line.decode().rstrip("\n").split("\t")[9])
        prior = md["prior_rater"]
        prior_errors = [md.get("prior_error")]
        prior_errors.extend(e.get("metadata", {}).get("prior_error") for e in md.get("deleted_errors", []))
        for err in prior_errors:
            if not err or err["severity"].lower() != "major":
                continue
            traces += 1
            if err["location"] not in {"source", "translation"}:
                raise ValueError(err["location"])
            side = "source_spans" if err["location"] == "source" else "target_spans"
            if originals[key, prior, side].isdisjoint(range(err["start"], err["end"]+1)):
                unmatched[key[1]] += 1
    if unmatched:
        raise ValueError(f"Check for artificial prior spans before round-two comparison: {unmatched}")
    return {"first_news_documents": len(first_docs), "second_documents": len(second_docs),
            "first_only_news_documents": sorted(first_docs-second_docs),
            "major_prior_trace_records": traces, "major_without_original_overlap": sum(unmatched.values())}


def records(cache):
    held = study.read(study.OUT/"seal.json")["split"]["held"]
    rows, jmat, human, masks = study.frame(cache, held)
    eager, gate = study.policies(jmat, human, masks)
    early_freq = (eager & ~gate).mean(axis=0)
    judgments = jmat.all(axis=1)
    truths = human.all(axis=2)
    framemap, _ = inputs._frame(cache)
    ids = {(k[1], f"{k[1]}|{k[2]}|{k[0]}"): (k, v[2])
           for k, v in framemap.items() if k[1] in held}
    second, _ = read_second(cache, {v[0] for v in ids.values()})
    out = []
    assignment = {}
    for i, r in enumerate(rows):
        key, trio = ids[r.doc, r.sid]
        priors = [second[(*key, name)][0] for name in trio]
        if sorted(priors) != sorted(trio) or sum(pr == who for pr, who in zip(priors,trio)) != 1:
            raise ValueError("Expected one self and two reciprocal other re-annotations")
        grouping = (tuple(trio), tuple(priors))
        if assignment.setdefault(r.doc, grouping) != grouping:
            raise ValueError("Changed rater assignment within document")
        for a, s in itertools.permutations(range(3), 2):
            prior, later = second[(*key,trio[s])]
            group = "self" if prior == trio[s] else ("auditor_prior" if prior == trio[a] else "third_person_prior")
            out.append(dict(doc=r.doc, unit=r.sid, a=trio[a], s=trio[s], slots=f"{a+1}->{s+1}",
                            judge=bool(judgments[i]), audit=bool(truths[i,a]), peer=bool(truths[i,s]),
                            later=later, early=float(early_freq[i,a]), seed_source=group))
    return out, assignment


def aggregate(rows, label, *, equal_outputs=True, ci=True):
    """Each output contributes unit mass, divided across eligible ordered roles."""
    units = Counter(r["unit"] for r in rows)
    groups = defaultdict(lambda: defaultdict(float))
    counts = Counter()
    for r in rows:
        j, a, p, later = r["judge"], r["audit"], r["peer"], r["later"]
        up, down = not j and a, j and not a
        e = r["early"]
        numbers = dict(n=1, j=j, a=a, p=p, up=up, down=down, early=e,
                       up_agree=up*p, down_agree=down*(not p), early_agree=e*p,
                       full=(a!=p), keep=(j!=p), later_full=(a!=later), later_keep=(j!=later),
                       later_up_agree=up*later, later_down_agree=down*(not later),
                       later_early_agree=e*later, later_peer=later,
                       pass_to_fail=(p and not later), fail_to_pass=(not p and later))
        weight = 1 / units[r["unit"]] if equal_outputs else 1
        for k, v in numbers.items():
            groups[r["doc"]][k] += weight*v
        counts["up_records"] += int(up)
        counts["down_records"] += int(down)
    docs = sorted(groups)
    fields = sorted(groups[docs[0]])
    totals = np.asarray([[groups[d][k] for k in fields] for d in docs])
    def calc(arr):
        v = dict(zip(fields, arr.T))
        n = v["n"]
        def pct(x, d):
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(d > 0, 100*x/d, np.nan)
        return dict(judge_pass=pct(v["j"],n), auditor_pass=pct(v["a"],n), peer_pass=pct(v["p"],n),
                    up=pct(v["up_agree"],v["up"]), down=pct(v["down_agree"],v["down"]),
                    any_change=pct(v["up_agree"]+v["down_agree"],v["up"]+v["down"]),
                    early=pct(v["early_agree"],v["early"]), early_per100=pct(v["early"],n),
                    keep=pct(v["keep"],n), full=pct(v["full"],n), diff=pct(v["full"]-v["keep"],n),
                    second_peer_pass=pct(v["later_peer"],n), second_full=pct(v["later_full"],n),
                    second_keep=pct(v["later_keep"],n), second_diff=pct(v["later_full"]-v["later_keep"],n),
                    shift=pct(v["later_full"]-v["later_keep"]-v["full"]+v["keep"],n),
                    second_up=pct(v["later_up_agree"],v["up"]),
                    second_down=pct(v["later_down_agree"],v["down"]),
                    second_early=pct(v["later_early_agree"],v["early"]),
                    pass_to_fail=pct(v["pass_to_fail"],n), fail_to_pass=pct(v["fail_to_pass"],n))
    centers = calc(totals.sum(axis=0))
    result = dict(label=label, documents=len(docs), outputs=len(units), role_records=len(rows), **counts)
    result.update({k:float(v) if np.isfinite(v) else None for k,v in centers.items()})
    if ci:
        # Single-document strata have no empirical between-document variation to estimate.
        if len(docs) < 2:
            return result
        weights = np.random.default_rng(20261125).multinomial(len(docs),np.ones(len(docs))/len(docs),size=1999)
        resampled = calc(weights @ totals)
        for k in ("up","down","any_change","early","diff","second_diff","shift"):
            available = resampled[k][np.isfinite(resampled[k])]
            result[k+"_valid"] = len(available)
            if len(available):
                result[k+"_lo"],result[k+"_hi"] = np.percentile(available,[2.5,97.5]).tolist()
    return result


def run(cache, destination):
    destination.mkdir(parents=True, exist_ok=True)
    rows, assignment = records(cache)
    audit = check_prior_spans(cache)
    all_people = sorted({r["a"] for r in rows},key=lambda r:int(r.removeprefix("rater")))
    total = aggregate(rows,"all")
    main = study.read(study.OUT/"held.json")["estimates"]
    for k, source in [("diff","full_minus_retain_1to1_per100"),("up","up_endorsement_pct"),
                      ("down","down_endorsement_pct"),("early","early_endorsement_pct")]:
        np.testing.assert_allclose([total[k],total[k+"_lo"],total[k+"_hi"]],
                                   [main[source]["point"],*main[source]["ci95"]],atol=1e-11)
    def by(key):
        subsets = defaultdict(list)
        for r in rows:
            subsets[key(r)].append(r)
        return [aggregate(group,label) for label,group in sorted(subsets.items())]
    slots = by(lambda r:r["slots"])
    actual = by(lambda r:r["a"]+"->"+r["s"])
    assert len(slots)==6 and all(r["outputs"]==total["outputs"] for r in slots)
    for group in (slots,actual):
        assert sum(g["role_records"] for g in group) == len(rows)
        np.testing.assert_allclose(np.average([g["diff"] for g in group],
                                              weights=[g["role_records"] for g in group]),total["diff"],atol=1e-10)
        for k in ("up","down"):
            denom = sum(g[k+"_records"] for g in group)
            assert denom>0
            expected = sum(g[k+"_records"]*(g[k] or 0) for g in group)/denom
            np.testing.assert_allclose(expected,total[k],atol=1e-10)
    omit = []
    person = []
    for who in all_people:
        kept = [r for r in rows if who not in (r["a"],r["s"])]
        result = aggregate(kept,who)
        influenced_docs = {d for d,(trio,_) in assignment.items() if who in trio}
        # Additional on-panel comparison holds document composition fixed before/after omitting a person.
        base = aggregate([r for r in rows if r["doc"] in influenced_docs],who,ci=False)
        influenced = aggregate([r for r in kept if r["doc"] in influenced_docs],who,ci=False)
        result.update(on_panel_outputs=base["outputs"],on_panel_docs=len(influenced_docs),
                      on_panel_original_diff=base["diff"],on_panel_without_diff=influenced["diff"],
                      on_panel_original_up=base["up"],on_panel_without_up=influenced["up"])
        assert result["outputs"]==total["outputs"]
        omit.append(result)
        person.append(aggregate([r for r in rows if r["a"]==who],who))
    rounds = by(lambda r:r["seed_source"])
    assert len(rounds)==3 and all(g["role_records"]==2*total["outputs"] for g in rounds)
    def first_pass_only(row):
        return {k:v for k,v in row.items() if not k.startswith(("second_","shift","pass_to_fail","fail_to_pass"))}
    def export_csv(path, data):
        columns = list(dict.fromkeys(k for r in data for k in r))
        with path.open('w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
            w.writeheader(); w.writerows(data)
    for name, data in (("pairs_slots",slots),("pairs_identities",actual),("pairs_omit",omit),("pairs_auditors",person)):
        export_csv(destination/(name+".csv"),[first_pass_only(x) for x in data])
    # One comparison per candidate expert against the other two, using agreement as the score.
    # The difference in mean tie-inclusive wins equals the raw accuracy difference on this binary panel.
    by_candidate = defaultdict(list)
    for r in rows:
        by_candidate[r["unit"],r["a"]].append(r)
    wins = Counter()
    for pair in by_candidate.values():
        assert len(pair)==2
        machine = sum(r["judge"]==r["peer"] for r in pair)
        candidate = sum(r["audit"]==r["peer"] for r in pair)
        wins["llm_strict"] += int(machine>candidate)
        wins["human_strict"] += int(machine<candidate)
        wins["ties"] += int(machine==candidate)
    n_candidates = len(by_candidate)
    alt = dict(candidate_output_roles=n_candidates, **wins,
               llm_at_least=(wins["llm_strict"]+wins["ties"])/n_candidates,
               human_at_least=(wins["human_strict"]+wins["ties"])/n_candidates)
    np.testing.assert_allclose(100*(alt["llm_at_least"]-alt["human_at_least"]),total["diff"],atol=1e-12)
    study.store(destination/"pairs_alt_identity.json",alt)
    study.store(destination/"pairs_summary.json",dict(original=first_pass_only(total),people=all_people,
                actual_directed_pairs=len(actual), slot_order="lexicographically sorted published rater ID within output",
                documents={d:list(pair[0]) for d,pair in sorted(assignment.items())},
                leave_one_out_weighting="exclude person in either role; renormalize remaining ordered pairs within each of the same outputs"))
    study.store(destination/"round2.json",dict(primary_sensitivity="third_person_prior",artificial_span_check=audit,
                comparison="first and second score of the same peer on the same primary judge and first-pass auditor",
                costs=[1,1],groups=rounds))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,default=study.ROOT/"artifacts/reports/wmt_followup_20261126")
    args=parser.parse_args()
    run(args.cache,args.out)
