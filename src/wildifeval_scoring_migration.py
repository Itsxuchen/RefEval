"""Offline, single-ledger migration for selected-prefix evidence serialization.

This command never accesses credentials, sends requests, reads reference labels,
changes existing job/attempt rows, or resets charges. It permits exactly one
configuration change: reducing the core cap from $100 to $99.50, with $0.50
reserved separately for unrelated diagnostic calls.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3

from src import wildifeval_scoring as s

MIGRATION_KIND = "selected-prefix-evidence-v2"


def _load(path):
    return json.loads(Path(path).read_text())


def _metadata(db):
    return dict(db.execute("SELECT key,value FROM metadata"))


def _fixed_logic(path):
    names = {"score_from_tokens", "request_body", "request_parameters", "usage_cost", "reservation_nano"}
    constants = {"PARSER_VERSION", "RATES", "MAX_INPUT_TOKENS", "SHORT_CONTEXT_TOKENS"}
    selected = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in constants:
                    selected[target.id] = ast.dump(node.value, include_attributes=False)
    if set(selected) != names | constants:
        raise s.GuardError("missing_fixed_logic_in_runner")
    return s.digest(selected)


def _immutable_digest(db):
    """Hash unchanged records, including all opaque stored P evidence."""
    import hashlib
    h = hashlib.sha256()
    for table, order in (("jobs", "job_id"), ("attempts", "id"), ("invocations", "id")):
        h.update(s.encoded(table))
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if exists:
            for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                h.update(s.encoded(list(row))+b"\n")
    row = db.execute("SELECT charge_nano,prior_nano,halted_reason FROM account WHERE id=1").fetchone()
    h.update(s.encoded(list(row)))
    return h.hexdigest()


def _assert_snapshot(db, path, success):
    if success:
        rows = db.execute("SELECT j.metadata_json,a.result_json,a.usage_json FROM jobs j JOIN attempts a ON j.job_id=a.job_id WHERE a.status='success' ORDER BY j.ordinal")
    else:
        rows = db.execute("SELECT metadata_json,status,attempt_count FROM jobs WHERE status<>'success' ORDER BY ordinal")
    count, terminal_ids, success_ids = 0, [], []
    with Path(path).open() as stream:
        for row in rows:
            meta = json.loads(row[0])
            expected = ({**meta, **json.loads(row[1]), "cost": json.loads(row[2])} if success
                        else {**meta, "status": row[1], "attempt_count": row[2]})
            line = stream.readline()
            if not line or s.encoded(json.loads(line)) != s.encoded(expected):
                raise s.GuardError("archived_export_does_not_match_ledger")
            count += 1
            if success:
                success_ids.append(meta["job_id"])
            elif row[1] in ("terminal_missing", "exhausted", "failed"):
                terminal_ids.append(meta["job_id"])
        if stream.readline():
            raise s.GuardError("archived_export_has_extra_rows")
    ids = sorted(success_ids if success else terminal_ids)
    return dict(rows=count, terminal_ids=ids, terminal_ids_sha256=s.digest(ids))


def _write_receipt(path, receipt):
    """Idempotent export; the committed SQLite row is authoritative."""
    path = Path(path)
    if path.exists():
        if s.encoded(_load(path)) != s.encoded(receipt):
            raise s.GuardError("migration_receipt_conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("wb") as handle:
        handle.write(s.encoded(receipt)+b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def migrate(*, jobs_path, old_config_path, new_config_path, state,
            old_runner_path, old_freeze_path, new_freeze_path, calibration_path,
            diagnostic_path, old_results_path, old_failures_path, backup_path,
            receipt_path):
    """Verify explicit predecessor artifacts, then migrate one existing ledger.

    A database backup is made through SQLite's backup API before mutation. A
    crash after commit can be recovered by rerunning these same arguments: the
    stored receipt is exported, without reapplying account or metadata changes.
    """
    old_config, new_config = _load(old_config_path), _load(new_config_path)
    s.validate_config(old_config)
    s.validate_config(new_config)
    if old_config.get("state") != "ready" or new_config.get("state") != "ready":
        raise s.GuardError("migration_requires_ready_configs")
    if (s.usd_to_nano(old_config["budget_usd"]) != 100*s.NANODOLLARS
            or s.usd_to_nano(new_config["budget_usd"]) != 99_500_000_000
            or {k:v for k,v in old_config.items() if k != "budget_usd"}
            != {k:v for k,v in new_config.items() if k != "budget_usd"}):
        raise s.GuardError("migration_may_only_reduce_budget_to_99_50")
    if s.file_hash(jobs_path) != new_config["jobs_sha256"]:
        raise s.GuardError("jobs_file_hash_mismatch")
    old_freeze, new_freeze = _load(old_freeze_path), _load(new_freeze_path)
    diagnostic = _load(diagnostic_path)
    if (diagnostic.get("source") != "unrelated_synthetic_logprob_compatibility"
            or s.usd_to_nano(diagnostic.get("budget_usd", -1)) > 500_000_000
            or type(diagnostic.get("charged_or_reserved_nano")) is not int
            or not 0 <= diagnostic["charged_or_reserved_nano"] <= 500_000_000
            or not isinstance(diagnostic.get("cases"), list)
            or len(diagnostic["cases"]) > 32):
        raise s.GuardError("invalid_unrelated_diagnostic_receipt")
    state = Path(state).resolve()
    if not (state/"ledger.sqlite").is_file():
        raise s.GuardError("existing_frozen_ledger_required")
    hashes = {"old_config_sha256": s.digest(old_config), "new_config_sha256": s.digest(new_config),
              "old_config_file_sha256": s.file_hash(old_config_path), "new_config_file_sha256": s.file_hash(new_config_path),
              "old_runner_sha256": s.file_hash(old_runner_path), "new_runner_sha256": s.file_hash(s.__file__),
              "old_generation_freeze_sha256": s.file_hash(old_freeze_path), "new_generation_freeze_sha256": s.file_hash(new_freeze_path),
              "calibration_sha256": s.file_hash(calibration_path), "diagnostic_sha256": s.file_hash(diagnostic_path),
              "old_results_sha256": s.file_hash(old_results_path), "old_failures_sha256": s.file_hash(old_failures_path),
              "migration_script_sha256": s.file_hash(__file__), "jobs_sha256": new_config["jobs_sha256"]}
    if hashes["old_runner_sha256"] == hashes["new_runner_sha256"]:
        raise s.GuardError("migration_requires_distinct_runner_versions")
    old_logic, new_logic = _fixed_logic(old_runner_path), _fixed_logic(s.__file__)
    if old_logic != new_logic:
        raise s.GuardError("migration_changed_score_request_or_pricing_logic")
    for freeze, config, runner in ((old_freeze, old_config, hashes["old_runner_sha256"]),
                                   (new_freeze, new_config, hashes["new_runner_sha256"])):
        if (freeze.get("status") != "frozen" or freeze.get("config_sha256") != s.digest(config)
                or freeze.get("runner_sha256") != runner
                or freeze.get("jobs_sha256") != config["jobs_sha256"]
                or freeze.get("calibration_sha256") != hashes["calibration_sha256"]
                or freeze.get("state_directory_sha256") != s.text_hash(str(state))):
            raise s.GuardError("invalid_migration_freeze")
    if (new_freeze.get("previous_generation_freeze_sha256") != hashes["old_generation_freeze_sha256"]
            or new_freeze.get("migration_script_sha256") != hashes["migration_script_sha256"]
            or new_freeze.get("compatibility_diagnostic_sha256") != hashes["diagnostic_sha256"]
            or new_freeze.get("evidence_serialization_version") != s.EVIDENCE_SERIALIZATION_VERSION
            or new_freeze.get("canary_job_ids_sha256") != old_freeze.get("canary_job_ids_sha256")
            or new_freeze.get("ledger_id") != old_freeze.get("ledger_id")):
        raise s.GuardError("migration_predecessor_chain_mismatch")
    # Reuse calibration checks without trying to open Ledger under old metadata.
    s.validate_execution_gates(new_config, calibration_path, new_freeze_path, state)
    migration_id = s.digest({**hashes, "ledger_id": new_freeze["ledger_id"], "kind": MIGRATION_KIND})
    lock = (state/"runner.lock").open("a+")
    db = None
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise s.GuardError("another_runner_owns_state") from None
        db = sqlite3.connect(state/"ledger.sqlite", isolation_level=None, timeout=60)
        db.execute("PRAGMA synchronous=FULL")
        meta = _metadata(db)
        if meta.get("ledger_id") != new_freeze["ledger_id"]:
            raise s.GuardError("frozen_ledger_identity_changed")
        has_table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='migrations'").fetchone()
        if has_table:
            existing = db.execute("SELECT receipt_json FROM migrations WHERE migration_id=?", (migration_id,)).fetchone()
            if existing:
                receipt = json.loads(existing[0])
                if (meta.get("config_sha256") != hashes["new_config_sha256"]
                        or meta.get("runner_sha256") != hashes["new_runner_sha256"]
                        or meta.get("generation_freeze_sha256") != hashes["new_generation_freeze_sha256"]
                        or s.file_hash(backup_path) != receipt["database_backup_sha256"]):
                    raise s.GuardError("committed_migration_state_changed")
                _write_receipt(receipt_path, receipt)
                return receipt
            if db.execute("SELECT COUNT(*) FROM migrations").fetchone()[0]:
                raise s.GuardError("different_migration_already_recorded")
        if (meta.get("config_sha256") != hashes["old_config_sha256"]
                or meta.get("config_json") != s.encoded(old_config).decode()
                or meta.get("runner_sha256") != hashes["old_runner_sha256"]
                or meta.get("generation_freeze_sha256") != hashes["old_generation_freeze_sha256"]
                or meta.get("calibration_sha256") != hashes["calibration_sha256"]):
            raise s.GuardError("ledger_predecessor_mismatch")
        if (db.execute("SELECT COUNT(*) FROM jobs WHERE status='inflight'").fetchone()[0]
                or db.execute("SELECT COUNT(*) FROM attempts WHERE status='inflight'").fetchone()[0]):
            raise s.GuardError("cannot_migrate_inflight_requests")
        account = db.execute("SELECT budget_nano,charge_nano,prior_nano,halted_reason FROM account WHERE id=1").fetchone()
        if (account[0] != s.usd_to_nano(old_config["budget_usd"]) or account[3] is not None
                or account[1] > s.usd_to_nano(new_config["budget_usd"])):
            raise s.GuardError("unsafe_account_for_migration")
        if db.execute("SELECT COALESCE(SUM(charged_nano),0) FROM attempts").fetchone()[0]+account[2] != account[1]:
            raise s.GuardError("ledger_charge_identity_failed")
        old_success = _assert_snapshot(db, old_results_path, True)
        old_failure = _assert_snapshot(db, old_failures_path, False)
        statuses = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        # Only the completed, fixed fee canary may precede this one-time repair.
        attempted = {row[0] for row in db.execute("SELECT job_id FROM jobs WHERE attempt_count>0")}
        groups = {}
        for job_id, metadata_json in db.execute("SELECT job_id,metadata_json FROM jobs"):
            groups.setdefault(json.loads(metadata_json)["producer"], []).append(job_id)
        expected = {j for ids in groups.values() for j in sorted(ids, key=lambda j:s.text_hash("fee-calibration-v1|"+j))[:32]}
        terminal = set(old_success["terminal_ids"]+old_failure["terminal_ids"])
        if attempted != expected or terminal != expected:
            raise s.GuardError("migration_requires_complete_original_canary_only")
        before = _immutable_digest(db)
        backup_path = Path(backup_path)
        if Path(receipt_path).exists():
            raise s.GuardError("migration_output_already_exists")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if backup_path.exists():
            # Recovery after backup but before commit: only reuse a complete,
            # matching predecessor snapshot. Never overwrite a partial or
            # unrelated backup to make a migration proceed.
            try:
                backup = sqlite3.connect(f"file:{backup_path.resolve()}?mode=ro", uri=True)
                try:
                    backup_account = backup.execute("SELECT budget_nano,charge_nano,prior_nano,halted_reason FROM account WHERE id=1").fetchone()
                    if (_metadata(backup) != meta or backup_account != account
                            or _immutable_digest(backup) != before
                            or backup.execute("PRAGMA quick_check").fetchone()[0] != "ok"):
                        raise s.GuardError("existing_backup_does_not_match_predecessor")
                finally:
                    backup.close()
            except sqlite3.DatabaseError:
                raise s.GuardError("existing_backup_does_not_match_predecessor") from None
        else:
            backup = sqlite3.connect(backup_path)
            try:
                db.backup(backup)
            finally:
                backup.close()
        backup_hash = s.file_hash(backup_path)
        receipt = dict(schema_version=1, status="completed", kind=MIGRATION_KIND,
                       migration_id=migration_id, created_utc=datetime.now(timezone.utc).isoformat(),
                       **hashes, ledger_id=new_freeze["ledger_id"], state_directory_sha256=s.text_hash(str(state)),
                       fixed_logic_ast_sha256=new_logic,
                       database_backup_sha256=backup_hash,
                       evidence_serialization_version=s.EVIDENCE_SERIALIZATION_VERSION,
                       old_success_jobs=old_success["rows"], old_terminal_failure_jobs=len(old_failure["terminal_ids"]),
                       old_success_ids=old_success["terminal_ids"], old_success_ids_sha256=old_success["terminal_ids_sha256"],
                       old_terminal_failure_ids=old_failure["terminal_ids"], old_terminal_failure_ids_sha256=old_failure["terminal_ids_sha256"],
                       job_status_counts=statuses, immutable_records_sha256=before,
                       charged_nano_unchanged=account[1], prior_nano_unchanged=account[2],
                       old_budget_nano=account[0], new_budget_nano=99_500_000_000,
                       diagnostic_budget_reserved_nano=500_000_000,
                       diagnostic_charged_or_reserved_nano=diagnostic["charged_or_reserved_nano"],
                       request_parameters_unchanged=True, score_algorithm_unchanged=True,
                       existing_results_unchanged=True, existing_missing_not_retried=True)
        db.execute("BEGIN IMMEDIATE")
        try:
            if _immutable_digest(db) != before:
                raise s.GuardError("ledger_changed_during_migration")
            db.execute("UPDATE account SET budget_nano=? WHERE id=1", (99_500_000_000,))
            for key, value in (("config_sha256", hashes["new_config_sha256"]), ("config_json", s.encoded(new_config).decode()),
                               ("runner_sha256", hashes["new_runner_sha256"]), ("generation_freeze_sha256", hashes["new_generation_freeze_sha256"]),
                               ("migration_id", migration_id)):
                db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, value))
            db.execute("CREATE TABLE IF NOT EXISTS migrations(migration_id TEXT PRIMARY KEY,receipt_json TEXT NOT NULL)")
            if _immutable_digest(db) != before:
                raise s.GuardError("migration_modified_immutable_records")
            db.execute("INSERT INTO migrations VALUES(?,?)", (migration_id, s.encoded(receipt).decode()))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        _write_receipt(receipt_path, receipt)
        return receipt
    finally:
        if db is not None:
            db.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("jobs", "old-config", "new-config", "state", "old-runner", "old-freeze", "new-freeze",
                 "calibration", "diagnostic", "old-results", "old-failures", "backup", "receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = migrate(**{("state" if k == "state" else k+"_path"):v for k,v in vars(args).items()})
        print(json.dumps({"status":receipt["status"], "migration_id":receipt["migration_id"],
                          "old_success_jobs":receipt["old_success_jobs"], "old_terminal_failure_jobs":receipt["old_terminal_failure_jobs"],
                          "api_calls":0}))
    except s.GuardError as exc:
        print(json.dumps({"status":"stopped", "reason":str(exc)}))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
