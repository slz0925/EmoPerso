"""Versioned, source-verified Pandora resolution overlay; never edits the base audit."""

import argparse
import hashlib
import itertools
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from .comment_index import (
    CORE_FIELDS, POLICY, _core, _source_span, fields_digest, normalize_comment_id,
    peak_memory_bytes,
)
from .errors import DataContractError
from .raw_data import (ReviewRequiredError, check_source, check_sqlite_snapshot,
                       load_users, pandora_user_key, read_span, write_json)

RESOLUTION_POLICY = "pandora-body-conflicts-v1"
RULES = {
    "no_core_conflict": "keep_first_original",
    "metadata_only": "keep_first_original",
    "whitespace_equivalent": "keep_first_original_without_rewriting",
    "non_whitespace_body_difference": "quarantine_entire_id_all_versions",
    "whitespace_comparison": "Unicode: ' '.join(body.split()) equality",
    "other_core_difference_or_invalid_record": "stop_without_finalizing",
}
EVIDENCE = ("manifest.json", "comment_audit.json", "conflict_review.jsonl", "conflict_impact.json")


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _db_stat(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _base(base):
    users, manifest = load_users(base, require_ready=False)
    if "resolution" in manifest:
        raise DataContractError("Resolution requires the original source audit, not another overlay")
    audit = (base / "comment_audit.json").read_bytes()
    if hashlib.sha256(audit).hexdigest() != manifest["comment_audit_sha256"]:
        raise DataContractError("Base audit digest mismatch")
    report = json.loads(audit)
    if (report["policy"] != POLICY or report["status"] not in ("ready", "review_required")
            or manifest["status"] != report["status"] or not report["source"]["complete"]
            or report["limit"] is not None or report["invalid_records"]):
        raise ReviewRequiredError("Resolution requires a complete audit with no unapproved invalid records")
    impact = json.loads((base / "conflict_impact.json").read_text(encoding="utf-8"))
    if (report["users_sha256"] != manifest["users_sha256"]
            or report["split_seed"] != manifest["seed"]
            or impact["comment_audit_sha256"] != manifest["comment_audit_sha256"]
            or impact["source_sha256"] != report["source"]["sha256"]
            or impact["review_sha256"] != file_sha(base / "conflict_review.jsonl")):
        raise DataContractError("Base review provenance mismatch")
    for source in (*manifest["sources"].values(), report["source"]):
        check_source(source)
    return users, manifest, report, impact


def _verify_versions(base, db, policy, report, impact, progress):
    """Re-read every duplicate and its baseline, including all 3+ version groups.

    Streaming groups and SQLite version lookups avoid an in-memory set of IDs or
    even an unbounded version set for a single pathological ID. The review JSONL
    is compared in lockstep, never treated as a replacement for source evidence.
    """
    counts, events, categories, multiplicity = Counter(), Counter(), Counter(), Counter()
    query = (
        "SELECT d.comment_id,d.kind,d.seen_version,d.core_fields,"
        "c.source_record,c.byte_start,c.byte_end,c.line_start,c.line_end,c.fields_sha,c.core_sha,"
        "c.duplicate_count,c.conflict_count,a.user_key,a.partition,"
        "d.source_record,d.byte_start,d.byte_end,d.line_start,d.line_end,d.fields_sha "
        "FROM duplicates d INDEXED BY duplicates_comment JOIN comments c USING(comment_id) "
        "JOIN authors a USING(author_id) ORDER BY d.comment_id,d.source_record")
    with check_source(report["source"]).open("rb") as source, (base / "conflict_review.jsonl").open("rb") as review:
        for identity, group in itertools.groupby(db.execute(query), key=lambda row: row[0]):
            original, duplicate_count, core_count, worst, has_white = None, 0, 0, 0, False
            for row in group:
                if original is None:
                    baseline = _source_span(row[4:10])
                    original = read_span(source, report["header"], baseline)
                    core = _core(original)
                    if (normalize_comment_id(original["id"]) != identity
                            or fields_digest(core) != row[10]
                            or pandora_user_key(original["author"]) != row[13]):
                        raise DataContractError("Baseline identity/core/author evidence mismatch")
                    counts["source_bytes_read"] += baseline.byte_end - baseline.byte_start
                    expected_duplicates, expected_core, user_key, partition = row[11:15]
                span = _source_span(row[15:21])
                variant = read_span(source, report["header"], span)
                counts["source_bytes_read"] += span.byte_end - span.byte_start
                if normalize_comment_id(variant["id"]) != identity or span.record <= baseline.record:
                    raise DataContractError("Duplicate identity/source order mismatch")
                changed = [name for name, a, b in zip(CORE_FIELDS, core, _core(variant)) if a != b]
                first = (baseline.record if span.fields_sha256 == baseline.fields_sha256 else
                         db.execute("SELECT first_record FROM variants WHERE comment_id=? AND fields_sha=?",
                                    (identity, bytes.fromhex(span.fields_sha256))).fetchone())
                if isinstance(first, tuple):
                    first = first[0]
                if first is None or first > span.record:
                    raise DataContractError("Missing full-field version evidence")
                seen = first < span.record
                kind = "core_conflict" if changed else "exact" if seen else "metadata_only"
                if (kind, int(seen), changed) != (row[1], row[2], json.loads(row[3])):
                    raise DataContractError("A duplicate version was omitted or incorrectly classified")
                counts[kind] += 1
                duplicate_count += 1
                counts["new_versions"] += not seen
                if changed:
                    core_count += 1
                    if changed != ["body"]:
                        raise ReviewRequiredError("Other core-field conflicts need a separate approved rule")
                    whitespace = " ".join(original["body"].split()) == " ".join(variant["body"].split())
                    has_white |= whitespace
                    worst = max(worst, 1 if whitespace else 2)
                    events["whitespace_equivalent" if whitespace else "non_whitespace_body_difference"] += 1
            if (duplicate_count, core_count) != (expected_duplicates, expected_core):
                raise DataContractError("Incomplete duplicate-ID version group")
            comparison = ("non_whitespace_body_difference" if worst == 2 else
                          "whitespace_equivalent" if worst == 1 else "no_core_conflict")
            if core_count:
                previous = json.loads(next(review, b"null"))
                expected = {"record_type": "conflicted_id", "comment_id_integer": identity,
                            "baseline_record": baseline.record, "baseline_user_key": user_key,
                            "baseline_partition": partition, "conflicting_records": core_count,
                            "comparison": comparison, "resolution": "unapproved"}
                if previous != expected:
                    raise DataContractError("Historical review omitted or misclassified an ID/version")
                categories[comparison] += 1
            multiplicity[duplicate_count + 1] += 1
            counts["mixed_whitespace_and_nonwhitespace_ids"] += has_white and worst == 2
            policy.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?)", (
                identity, "quarantine" if worst == 2 else "keep_first", comparison,
                duplicate_count + 1, baseline.record, core_count))
            counts["duplicated_ids"] += 1
            if counts["duplicated_ids"] % 10000 == 0:
                policy.commit()
                if progress:
                    progress({"ids_verified": counts["duplicated_ids"],
                              "duplicate_versions_verified": sum(counts[k] for k in ("exact", "metadata_only", "core_conflict"))})
        if next(review, None) is not None:
            raise DataContractError("Historical review has unmatched records")
    if ((counts["exact"], counts["metadata_only"], counts["core_conflict"], counts["duplicated_ids"]) !=
            (report["exact_duplicates"], report["metadata_variants"], report["core_conflicts"], report["duplicated_ids"])
            or dict(events) != impact["conflict_record_categories"]
            or dict(categories) != impact["conflicted_id_categories"]
            or counts["new_versions"] != db.execute("SELECT COUNT(*) FROM variants").fetchone()[0]):
        raise DataContractError("Full version/review accounting mismatch")
    policy.commit()
    return {"status": "all_duplicate_versions_verified", "counts": dict(counts),
            "versions_per_id": dict(sorted(multiplicity.items())),
            "conflict_record_categories": dict(events), "conflicted_id_categories": dict(categories),
            "source_records_read": counts["duplicated_ids"] + report["duplicate_records"],
            "unique_nonduplicate_records": "reused completed base index; not rescanned"}


def resolve_conflicts(base, output, *, progress=None):
    """Create a new overlay only after all source-backed version checks succeed.

    On failure its manifest remains review_required; the original is never edited.
    Baselines and all variants remain addressable through the original audit.
    """
    start = time.perf_counter()
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Resolution output already exists; preserve previous results")
    users, manifest, report, impact = _base(base)
    evidence = {name: file_sha(base / name) for name in EVIDENCE}
    base_stat = _db_stat(base / "comments.sqlite")
    free_before = shutil.disk_usage(output.parent).free
    if free_before < (base / "users.jsonl").stat().st_size + 1024**3:
        raise OSError("Insufficient disk reserve for resolution overlay")
    output.mkdir(exist_ok=False)
    shutil.copyfile(base / "users.jsonl", output / "users.jsonl")
    pending = dict(manifest, status="review_required")
    pending["resolution"] = {"policy": RESOLUTION_POLICY, "base_directory": str(base),
                             "state": "verification_pending"}
    write_json(output / "manifest.json", pending)
    with closing(sqlite3.connect((base / "comments.sqlite").as_uri()+"?mode=ro", uri=True)) as db, \
            closing(sqlite3.connect(output / "resolution.sqlite")) as policy:
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA query_only=ON")
        policy.execute("PRAGMA cache_size=-32768")
        policy.execute("PRAGMA temp_store=FILE")
        policy.executescript("""
            CREATE TABLE decisions(comment_id INTEGER PRIMARY KEY, action TEXT NOT NULL,
              comparison TEXT NOT NULL, version_count INTEGER NOT NULL,
              baseline_record INTEGER NOT NULL, core_conflict_count INTEGER NOT NULL);
            CREATE TABLE user_impact(user_key TEXT PRIMARY KEY, mbti TEXT, partition TEXT,
              label_eligible INTEGER NOT NULL, profile_exists INTEGER NOT NULL,
              original_comments INTEGER NOT NULL, excluded_comments INTEGER NOT NULL,
              remaining_comments INTEGER NOT NULL);
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        if json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0]) != report:
            raise DataContractError("Base index metadata mismatch")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise DataContractError("Base index integrity check failed")
        ledger = db.execute("SELECT COUNT(*),SUM(duplicate_count),SUM(conflict_count),SUM(conflict_count>0) "
                            "FROM comments").fetchone()
        if ledger != (report["unique_comments"], report["duplicate_records"], report["core_conflicts"], report["conflicted_ids"]):
            raise DataContractError("Base index ledger mismatch")
        if db.execute("SELECT COUNT(*) FROM invalid_records").fetchone()[0]:
            raise ReviewRequiredError("Invalid records require a separate rule")
        verification = _verify_versions(base, db, policy, report, impact, progress)
        write_json(output / "verification.json", verification)
        policy.execute("CREATE INDEX decisions_action ON decisions(action,comment_id)")
        policy.commit()
        # Only the bounded profile roster and per-author counts are materialized.
        profiles = {u.user_key: u for u in users if u.dataset_id == "pandora"}
        authors = db.execute("SELECT author_id,user_key,profile_exists,label_eligible,partition FROM authors").fetchall()
        reference = {u.user_key: (1, int(u.label_eligible), u.partition) for u in profiles.values()}
        if {r[1]: r[2:] for r in authors if r[2]} != reference:
            raise DataContractError("Source index author/label/partition associations differ")
        totals = dict(db.execute("SELECT author_id,COUNT(*) FROM comments INDEXED BY comments_author GROUP BY author_id"))
        excluded = Counter()
        for (identity,) in policy.execute("SELECT comment_id FROM decisions WHERE action='quarantine'"):
            found = db.execute("SELECT author_id FROM comments WHERE comment_id=?", (identity,)).fetchone()
            excluded[found[0]] += 1
        for author_id, key, exists, eligible, partition in authors:
            user = profiles.get(key)
            policy.execute("INSERT INTO user_impact VALUES (?,?,?,?,?,?,?,?)", (
                key, user.mbti if user else None, partition, eligible, exists,
                totals.get(author_id, 0), excluded[author_id], totals.get(author_id, 0)-excluded[author_id]))
        policy.commit()
        summary = {"policy": RESOLUTION_POLICY, "rules": RULES, "status": "ready",
                   "source_sha256": report["source"]["sha256"], "users_sha256": manifest["users_sha256"],
                   "unique_comment_ids": report["unique_comments"], "excluded_unique_ids": sum(excluded.values()),
                   "retained_unique_ids": report["unique_comments"]-sum(excluded.values()),
                   "affected_profile_users": policy.execute("SELECT COUNT(*) FROM user_impact WHERE profile_exists=1 AND excluded_comments>0").fetchone()[0],
                   "affected_supervised_users": policy.execute("SELECT COUNT(*) FROM user_impact WHERE label_eligible=1 AND excluded_comments>0").fetchone()[0],
                   "profile_users_with_remaining_comments": policy.execute("SELECT COUNT(*) FROM user_impact WHERE profile_exists=1 AND remaining_comments>0").fetchone()[0],
                   "supervised_users_with_remaining_comments": policy.execute("SELECT COUNT(*) FROM user_impact WHERE label_eligible=1 AND remaining_comments>0").fetchone()[0],
                   "supervised_users_without_remaining_comments": policy.execute("SELECT COUNT(*) FROM user_impact WHERE label_eligible=1 AND remaining_comments=0").fetchone()[0],
                   "by_partition": {}, "by_mbti": {}, "sample_eligibility": "not_assessed"}
        for column, target in (("partition", "by_partition"), ("mbti", "by_mbti")):
            for value, n, affected, before, removed, remaining, zero in policy.execute(
                    f"SELECT {column},COUNT(*),SUM(excluded_comments>0),SUM(original_comments),"
                    f"SUM(excluded_comments),SUM(remaining_comments),SUM(remaining_comments=0) "
                    f"FROM user_impact GROUP BY {column}"):
                summary[target][value or "unassigned"] = dict(users=n, affected_users=affected,
                    original_comments=before, excluded_comments=removed, remaining_comments=remaining,
                    users_without_comments=zero)
        policy.execute("INSERT INTO metadata VALUES ('policy',?)", (RESOLUTION_POLICY,))
        policy.execute("INSERT INTO metadata VALUES ('summary',?)", (json.dumps(summary, sort_keys=True),))
        policy.commit()
        if policy.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise DataContractError("Resolution database integrity check failed")
    # Recheck every bound input before publishing readiness; never touch base files.
    if evidence != {name: file_sha(base / name) for name in EVIDENCE} or base_stat != _db_stat(base / "comments.sqlite"):
        raise DataContractError("Historical evidence changed during resolution")
    for source in (*manifest["sources"].values(), report["source"]):
        check_source(source)
    if file_sha(output / "users.jsonl") != manifest["users_sha256"]:
        raise DataContractError("User splits changed during resolution")
    summary.update(elapsed_seconds=time.perf_counter()-start, peak_memory_bytes=peak_memory_bytes(),
                   free_disk_before=free_before, free_disk_after=shutil.disk_usage(output).free,
                   python=sys.version, sqlite=sqlite3.sqlite_version,
                   code_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                                ("conflict_resolution.py", "comment_index.py", "raw_data.py")})
    write_json(output / "resolution_report.json", summary)
    pending.update(status="ready", resolution={"policy": RESOLUTION_POLICY, "rules": RULES,
        "state": "all_versions_verified_and_resolved", "base_directory": str(base),
        "base_evidence_sha256": evidence, "base_index_stat": base_stat,
        "database_sha256": file_sha(output / "resolution.sqlite"),
        "verification_sha256": file_sha(output / "verification.json"),
        "report_sha256": file_sha(output / "resolution_report.json")})
    write_json(output / "manifest.json", pending)
    return summary


def validate_resolution(output, manifest):
    """Validate an overlay before user or comment access; pending bases stay blocked."""
    output = Path(output).resolve()
    resolution = manifest["resolution"]
    if (manifest["status"] != "ready" or resolution.get("policy") != RESOLUTION_POLICY
            or resolution.get("rules") != RULES
            or resolution.get("state") != "all_versions_verified_and_resolved"):
        raise ReviewRequiredError("Resolution has not passed the approved policy")
    base = Path(resolution["base_directory"]).resolve()
    check_sqlite_snapshot(base / "comments.sqlite")
    check_sqlite_snapshot(output / "resolution.sqlite")
    for name, expected in resolution["base_evidence_sha256"].items():
        if name not in EVIDENCE or file_sha(base / name) != expected:
            raise DataContractError("Historical evidence digest mismatch")
    if set(resolution["base_evidence_sha256"]) != set(EVIDENCE):
        raise DataContractError("Incomplete resolution provenance")
    for name, key in (("resolution.sqlite", "database_sha256"), ("verification.json", "verification_sha256"),
                      ("resolution_report.json", "report_sha256")):
        if file_sha(output / name) != resolution[key]:
            raise DataContractError("Resolution artifact digest mismatch")
    _, original = load_users(base, require_ready=False)
    if any(manifest[k] != original[k] for k in original if k != "status"):
        raise DataContractError("Overlay changed base data, user split or source metadata")
    if _db_stat(base / "comments.sqlite") != resolution["base_index_stat"]:
        raise DataContractError("Base index was modified")
    for source in (*original["sources"].values(), original["comment_source"]):
        check_source(source)
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    last = [0.0]

    def progress(info):
        now = time.monotonic()
        if now-last[0] >= 30:
            print(json.dumps(info), flush=True)
            last[0] = now

    result = resolve_conflicts(args.base, args.output, progress=progress)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
