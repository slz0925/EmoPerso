"""Compare conflicting source versions without cleaning or selecting training text."""

import argparse
import hashlib
import itertools
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .comment_index import CORE_FIELDS, _core, _source_span, peak_memory_bytes
from .errors import DataContractError
from .raw_data import ReviewRequiredError, SourceSpan, check_source, load_users, pandora_user_key, read_span, write_json

COMPARISON_POLICY = "audit-only-v1: exact Unicode whitespace split/join comparison; no output text transformation"


def audit_conflicts(output, *, progress=None):
    """Stream every core-conflict pair and invalid source row from a completed audit.

    ID groups are read in indexed order, so no all-comment-ID set is allocated.
    Whitespace equivalence is a diagnostic, not permission to merge versions.
    New private review files never change the bundle's readiness or source index.
    """
    start = time.perf_counter()
    output = Path(output).resolve()
    users, manifest = load_users(output, require_ready=False)
    raw_audit = (output / "comment_audit.json").read_bytes()
    if hashlib.sha256(raw_audit).hexdigest() != manifest["comment_audit_sha256"]:
        raise DataContractError("Comment audit digest mismatch")
    report = json.loads(raw_audit)
    if not report["source"]["complete"]:
        raise ReviewRequiredError("Content review requires a completed source scan")
    if report["users_sha256"] != manifest["users_sha256"]:
        raise DataContractError("User manifest and source index differ")
    review_path, summary_path = output / "conflict_review.jsonl", output / "conflict_impact.json"
    if review_path.exists() or summary_path.exists():
        raise FileExistsError("Conflict review already exists; do not overwrite audit evidence")
    source_path = check_source(report["source"])
    profiles = {u.user_key: u for u in users if u.dataset_id == "pandora"}
    affected = set()  # At most the known profile roster, not the comment-ID population.
    events, identities, fields, invalid = Counter(), Counter(), Counter(), Counter()
    levels = ("whitespace_equivalent", "non_whitespace_body_difference", "other_core_difference")
    db = sqlite3.connect((output / "comments.sqlite").as_uri() + "?mode=ro", uri=True)
    digest = hashlib.sha256()
    processed = 0
    try:
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA temp_store=FILE")
        stored = json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
        if stored != report:
            raise DataContractError("Index report and audit file differ")
        rows = db.execute(
            "SELECT d.comment_id,a.user_key,a.partition,c.source_record,c.byte_start,c.byte_end,c.line_start,"
            "c.line_end,c.fields_sha,d.source_record,d.byte_start,d.byte_end,d.line_start,d.line_end,"
            "d.fields_sha,d.core_fields FROM duplicates d INDEXED BY duplicates_comment "
            "JOIN comments c USING(comment_id) JOIN authors a USING(author_id) "
            "WHERE d.kind='core_conflict' ORDER BY d.comment_id,d.source_record")
        with source_path.open("rb") as source, review_path.open("xb") as review:
            for identity, group in itertools.groupby(rows, key=lambda row: row[0]):
                worst, count, original = 0, 0, None
                for row in group:
                    if original is None:
                        baseline = _source_span(row[3:9])
                        original = read_span(source, report["header"], baseline)
                        baseline_user, partition = row[1], row[2]
                    variant = read_span(source, report["header"], _source_span(row[9:15]))
                    changed = [name for name, a, b in zip(CORE_FIELDS, _core(original), _core(variant)) if a != b]
                    if changed != json.loads(row[15]):
                        raise DataContractError("Stored core-conflict evidence does not match the source")
                    fields.update(changed)
                    if changed != ["body"]:
                        level = 2
                    elif " ".join(original["body"].split()) == " ".join(variant["body"].split()):
                        level = 0
                    else:
                        level = 1
                    worst, count = max(worst, level), count + 1
                    events[levels[level]] += 1
                    processed += 1
                    if progress and processed % 5000 == 0:
                        progress({"conflict_records_reviewed": processed,
                                  "seconds": round(time.perf_counter()-start, 2),
                                  "peak_memory_bytes": peak_memory_bytes()})
                    for author in (original["author"], variant["author"]):
                        key = pandora_user_key(author)
                        if key in profiles:
                            affected.add(key)
                identities[levels[worst]] += 1
                value = {"record_type": "conflicted_id", "comment_id_integer": identity,
                         "baseline_record": baseline.record, "baseline_user_key": baseline_user,
                         "baseline_partition": partition, "conflicting_records": count,
                         "comparison": levels[worst], "resolution": "unapproved"}
                raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                review.write(raw)
                digest.update(raw)
            for row in db.execute("SELECT source_record,byte_start,byte_end,line_start,line_end,fields_sha "
                                  "FROM invalid_records ORDER BY source_record"):
                span = SourceSpan(*row)
                raw = read_span(source, report["header"], span)
                category = ("repeated_header" if [raw[h] for h in report["header"]] == report["header"]
                            else "other_invalid_identity")
                invalid[category] += 1
                value = {"record_type": "invalid_source_record", "source_record": span.record,
                         "comparison": category, "resolution": "unapproved"}
                encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                review.write(encoded)
                digest.update(encoded)
        check_source(report["source"])
        if (sum(events.values()), sum(identities.values()), sum(invalid.values())) != (
                report["core_conflicts"], report["conflicted_ids"], report["invalid_records"]):
            raise DataContractError("Conflict review accounting mismatch")
        supervised = [profiles[key] for key in affected if profiles[key].label_eligible]
        summary = {"status": "audit_only_unapproved", "comparison_policy": COMPARISON_POLICY,
                   "source_sha256": report["source"]["sha256"],
                   "comment_audit_sha256": manifest["comment_audit_sha256"],
                   "review_sha256": digest.hexdigest(), "conflict_record_categories": dict(events),
                   "conflicted_id_categories": dict(identities), "core_field_counts": dict(fields),
                   "invalid_record_categories": dict(invalid), "affected_profile_users": len(affected),
                   "affected_supervised_users": len(supervised),
                   "affected_supervised_partitions": dict(Counter(u.partition for u in supervised)),
                   "elapsed_seconds": time.perf_counter()-start, "peak_memory_bytes": peak_memory_bytes(),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write_json(summary_path, summary)
        return summary
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    previous = [0.0]

    def progress(info):
        if info["seconds"] - previous[0] >= 30:
            print(json.dumps(info), flush=True)
            previous[0] = info["seconds"]

    print(json.dumps(audit_conflicts(args.output, progress=progress), sort_keys=True))


if __name__ == "__main__":
    main()
