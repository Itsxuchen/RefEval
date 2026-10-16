"""Synthetic-only migration: no credentials, network, or reference outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from src import wildifeval_scoring as s
from src import wildifeval_scoring_migration as m


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def fixture(tmp, n=4):
    jobs_path = tmp/"jobs.jsonl"
    rows = []
    for i in range(n):
        prompt = "Unrelated fixture " + str(i)
        criterion, h = "criterion-"+str(i), s.text_hash(prompt)
        rows.append(dict(job_id=s.text_hash("wildifeval-new-p-v1|"+criterion+"|"+h),
                         unit_id="u"+str(i), task_id="t"+str(i), cluster_id="c"+str(i),
                         producer="p"+str(i%2), criterion_index=0, criterion_id=criterion,
                         k=1, split="pilot", prompt=prompt, prompt_sha256=h,
                         output_hash=s.text_hash("output"+str(i)), criterion_hash=s.text_hash(criterion)))
    jobs_path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    config = s.config_example(s.file_hash(jobs_path))
    config.update(state="ready", requests_per_second=100, concurrency=2)
    state = tmp/"state"
    ledger = s.Ledger(state, config, jobs_path)
    identity = ledger.summary()["ledger_id"]
    ledger.close()
    calibration = write(tmp/"calibration.json", dict(status="passed", source="unrelated_synthetic_probe",
        model_requested=config["model"], response_model=config["model"], system_fingerprint=None,
        request_parameters=s.request_parameters(config), usage_bound_ok=True, score_parse_ok=True))
    canary = tmp/"canary.json"
    s.make_canary_manifest(jobs_path, canary)
    old_freeze = dict(status="frozen", config_sha256=s.digest(config), jobs_sha256=config["jobs_sha256"],
        calibration_sha256=s.file_hash(calibration), runner_sha256=s.file_hash(s.__file__), ledger_id=identity,
        state_directory_sha256=s.text_hash(str(state.resolve())), canary_job_ids_sha256=s.file_hash(canary))
    old_freeze_path = write(tmp/"old_freeze.json", old_freeze)
    count = [0]
    def request(_):
        count[0] += 1
        return dict(id="synthetic", model="gpt-6-luna", service_tier="default",
                    usage=dict(prompt_tokens=10,completion_tokens=1), choices=[dict(
                        finish_reason="length" if count[0] == 1 else "stop",
                        message=dict(content="Yes"), logprobs=dict(content=[dict(top_logprobs=[dict(token="Yes",logprob=-.1)])]))])
    s.run_jobs(jobs_path, config, state, request, calibration_receipt=calibration,
               generation_freeze=old_freeze_path, job_ids_path=canary, phase_cap_usd=1)
    # Represent immutable v1 rows. This fixture changes no real run or artifacts.
    ledger = s.Ledger(state, config, jobs_path, require_existing=True)
    for aid, result_json in list(ledger.db.execute("SELECT id,result_json FROM attempts WHERE status='success'")):
        row = json.loads(result_json)
        row.pop("evidence_serialization_version", None)
        row.pop("invalid_prefix_positions", None)
        ledger.db.execute("UPDATE attempts SET result_json=? WHERE id=?", (s.encoded(row).decode(),aid))
    ledger.db.execute("UPDATE attempts SET error_code='malformed_logprobs' WHERE status='terminal_missing'")
    ledger.export()
    old_results, old_failures = tmp/"old_results.jsonl", tmp/"old_failures.jsonl"
    old_results.write_bytes((state/"results.jsonl").read_bytes())
    old_failures.write_bytes((state/"failures.jsonl").read_bytes())
    old_runner = tmp/"old_scoring.py"
    old_runner.write_text(Path(s.__file__).read_text()+"\n# synthetic predecessor snapshot\n")
    old_freeze["runner_sha256"] = s.file_hash(old_runner)
    write(old_freeze_path, old_freeze)
    ledger.db.execute("UPDATE metadata SET value=? WHERE key='runner_sha256'", (s.file_hash(old_runner),))
    ledger.db.execute("UPDATE metadata SET value=? WHERE key='generation_freeze_sha256'", (s.file_hash(old_freeze_path),))
    ledger.close()
    old_config = write(tmp/"old_config.json", config)
    new_config = {**config,"budget_usd":99.5}
    new_config_path = write(tmp/"new_config.json", new_config)
    diagnostic = write(tmp/"diagnostic.json", dict(source="unrelated_synthetic_logprob_compatibility",
                         budget_usd=.5, charged_or_reserved_nano=484000, cases=[dict(status="returned")]))
    new_freeze = {**old_freeze, "config_sha256":s.digest(new_config), "runner_sha256":s.file_hash(s.__file__),
                  "previous_generation_freeze_sha256":s.file_hash(old_freeze_path), "migration_script_sha256":s.file_hash(m.__file__),
                  "compatibility_diagnostic_sha256":s.file_hash(diagnostic), "evidence_serialization_version":s.EVIDENCE_SERIALIZATION_VERSION}
    new_freeze_path = write(tmp/"new_freeze.json", new_freeze)
    return dict(jobs_path=jobs_path, old_config_path=old_config, new_config_path=new_config_path, state=state,
                old_runner_path=old_runner, old_freeze_path=old_freeze_path, new_freeze_path=new_freeze_path,
                calibration_path=calibration, diagnostic_path=diagnostic, old_results_path=old_results,
                old_failures_path=old_failures, backup_path=tmp/"before.sqlite", receipt_path=tmp/"migration.json")


def test_migration_preserves_success_missing_attempts_and_charges(tmp_path):
    args = fixture(tmp_path, n=66)
    before_results = args["old_results_path"].read_bytes()
    db = sqlite3.connect(args["state"]/"ledger.sqlite")
    before = m._immutable_digest(db)
    db.close()
    receipt = m.migrate(**args)
    assert receipt["old_success_jobs"] == 63 and receipt["old_terminal_failure_jobs"] == 1
    assert receipt["job_status_counts"] == {"success":63,"terminal_missing":1,"pending":2}
    assert receipt["new_budget_nano"] == 99_500_000_000
    assert receipt["diagnostic_budget_reserved_nano"] == 500_000_000
    config = m._load(args["new_config_path"])
    ledger = s.Ledger(args["state"], config, args["jobs_path"], require_existing=True)
    assert m._immutable_digest(ledger.db) == before
    assert ledger.summary()["committed_or_unknown_usd"] == receipt["charged_nano_unchanged"]/s.NANODOLLARS
    ledger.export()
    assert (args["state"]/"results.jsonl").read_bytes() == before_results
    pending = ledger.db.execute("SELECT job_id FROM jobs WHERE status='pending'").fetchall()
    claim, reason = ledger.claim()
    assert reason == "claimed" and claim["job_id"] in {row[0] for row in pending}
    ledger.finish(claim, cancelled=True)
    ledger.close()
    assert m.migrate(**args) == receipt


def test_post_commit_receipt_crash_replay_is_idempotent(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    writer = m._write_receipt
    monkeypatch.setattr(m, "_write_receipt", lambda *a: (_ for _ in ()).throw(OSError("synthetic crash")))
    with pytest.raises(OSError):
        m.migrate(**args)
    assert not args["receipt_path"].exists() and args["backup_path"].exists()
    db = sqlite3.connect(args["state"]/"ledger.sqlite")
    after = m._immutable_digest(db)
    assert db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 1
    db.close()
    monkeypatch.setattr(m, "_write_receipt", writer)
    result = m.migrate(**args)
    assert args["receipt_path"].exists()
    db = sqlite3.connect(args["state"]/"ledger.sqlite")
    assert m._immutable_digest(db) == after
    assert db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0] == 1
    db.close()
    assert m.migrate(**args) == result


@pytest.mark.parametrize("damage", ["config", "source", "freeze", "snapshot", "inflight", "halt", "charge", "diagnostic", "active"])
def test_reject_unsafe_migration_before_mutation(tmp_path, damage):
    args = fixture(tmp_path)
    lock = None
    db = sqlite3.connect(args["state"]/"ledger.sqlite")
    if damage == "config":
        value=m._load(args["new_config_path"]);value["max_attempts"]=2;write(args["new_config_path"],value)
    elif damage == "source":
        value=args["old_runner_path"].read_text().replace('total > .0001','total > .001');args["old_runner_path"].write_text(value)
    elif damage == "freeze":
        value=m._load(args["new_freeze_path"]);value["previous_generation_freeze_sha256"]="0"*64;write(args["new_freeze_path"],value)
    elif damage == "snapshot":
        args["old_results_path"].write_text("")
    elif damage == "inflight":
        db.execute("UPDATE jobs SET status='inflight' WHERE ordinal=0")
    elif damage == "halt":
        db.execute("UPDATE account SET halted_reason='insufficient_quota'")
    elif damage == "charge":
        db.execute("UPDATE account SET charge_nano=charge_nano+1")
    elif damage == "diagnostic":
        value=m._load(args["diagnostic_path"]);value["charged_or_reserved_nano"]=500000001;write(args["diagnostic_path"],value)
    elif damage == "active":
        import fcntl
        lock=(args["state"]/"runner.lock").open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db.commit()
    before=m._immutable_digest(db)
    db.close()
    try:
        with pytest.raises(s.GuardError):
            m.migrate(**args)
        assert not args["backup_path"].exists() and not args["receipt_path"].exists()
        db=sqlite3.connect(args["state"]/"ledger.sqlite")
        assert m._immutable_digest(db)==before
        assert db.execute("SELECT budget_nano FROM account").fetchone()[0]==100*s.NANODOLLARS
        db.close()
    finally:
        if lock is not None:
            lock.close()


def test_precommit_crash_reuses_only_matching_backup(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = m._immutable_digest
    calls = [0]
    def crash_after_backup(db):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("synthetic precommit crash")
        return original(db)
    monkeypatch.setattr(m, "_immutable_digest", crash_after_backup)
    with pytest.raises(RuntimeError):
        m.migrate(**args)
    assert args["backup_path"].exists() and not args["receipt_path"].exists()
    backup_hash = s.file_hash(args["backup_path"])
    monkeypatch.setattr(m, "_immutable_digest", original)
    receipt = m.migrate(**args)
    assert receipt["database_backup_sha256"] == backup_hash
    assert s.file_hash(args["backup_path"]) == backup_hash


def test_unrelated_existing_backup_is_never_overwritten(tmp_path):
    args = fixture(tmp_path)
    args["backup_path"].write_bytes(b"unrelated bytes")
    before = args["backup_path"].read_bytes()
    with pytest.raises(s.GuardError, match="existing_backup_does_not_match_predecessor"):
        m.migrate(**args)
    assert args["backup_path"].read_bytes() == before
    db = sqlite3.connect(args["state"]/"ledger.sqlite")
    assert db.execute("SELECT budget_nano FROM account").fetchone()[0] == 100*s.NANODOLLARS
    db.close()
