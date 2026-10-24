"""Bounded-memory Pandora identity audit and source-backed SQLite comment index.

The index records source locations and duplicate versions; comment bodies
remain in the original CSV. Reads preserve source order and original text.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DataContractError
from .raw_data import (
    CsvSource, MAX_RECORD_BYTES, ReviewRequiredError, SourceSpan, check_source, check_sqlite_snapshot,
    fields_digest, load_users, pandora_user_key, prepare_users, read_span, write_json,
)

CORE_FIELDS = ("author", "body", "created_utc", "link_id", "parent_id", "subreddit_id")
REQUIRED_FIELDS = CORE_FIELDS + ("id",)
POLICY = "first-logical-record; all-field-variants-tracked; unresolved-core-conflicts-block-use-v1"


def normalize_comment_id(value):
    """Canonical nonnegative base36 identity, optionally a Reddit t1_ fullname."""
    if not isinstance(value, str):
        raise ValueError("Comment ID must be a string")
    value = value.strip().lower()
    if value.startswith("t1_"):
        value = value[3:]
    if not re.fullmatch(r"[0-9a-z]+", value):
        raise ValueError("Invalid comment ID")
    value = value.lstrip("0") or "0"
    if len(value) > 13:
        raise ValueError("Comment ID exceeds SQLite integer range")
    number = int(value, 36)
    if number > 2**63 - 1:
        raise ValueError("Comment ID exceeds SQLite integer range")
    return number


def normalize_timestamp(value):
    # Avoid floating-point rounding; 1, 1.0 and 01.000 represent the same UTC time.
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        raise ValueError("Invalid nonnegative decimal UTC timestamp")
    whole, _, fraction = value.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return whole + ("." + fraction if fraction else "")


def _core(row):
    return tuple(normalize_timestamp(row[name]) if name == "created_utc" else row[name]
                 for name in CORE_FIELDS)


def peak_memory_bytes():
    """OS process high-water RSS/working set, not Python allocations alone."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.peak
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


SCHEMA = """
CREATE TABLE authors (
 author_id INTEGER PRIMARY KEY, author TEXT NOT NULL UNIQUE,
 user_key TEXT NOT NULL UNIQUE, profile_exists INTEGER NOT NULL,
 label_eligible INTEGER NOT NULL, partition TEXT
);
CREATE TABLE comments (
 comment_id INTEGER PRIMARY KEY, author_id INTEGER NOT NULL REFERENCES authors,
 source_record INTEGER NOT NULL, byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
 line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
 core_sha BLOB NOT NULL, fields_sha BLOB NOT NULL,
 duplicate_count INTEGER NOT NULL DEFAULT 0, conflict_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE variants (
 comment_id INTEGER NOT NULL REFERENCES comments, fields_sha BLOB NOT NULL,
 first_record INTEGER NOT NULL, PRIMARY KEY(comment_id, fields_sha)
) WITHOUT ROWID;
CREATE TABLE duplicates (
 source_record INTEGER PRIMARY KEY, comment_id INTEGER NOT NULL REFERENCES comments,
 byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
 line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, fields_sha BLOB NOT NULL,
 kind TEXT NOT NULL, seen_version INTEGER NOT NULL, core_fields TEXT NOT NULL
);
CREATE TABLE invalid_records (
 source_record INTEGER PRIMARY KEY, byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
 line_start INTEGER NOT NULL, line_end INTEGER NOT NULL, fields_sha TEXT NOT NULL,
 reason TEXT NOT NULL
);
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _put_metadata(db, report):
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('report', ?)",
               (json.dumps(report, sort_keys=True),))
    db.commit()


def _source_span(row):
    return SourceSpan(row[0], row[1], row[2], row[3], row[4], row[5].hex())


def build_comment_index(source_path, output, *, limit=None, batch_size=20000,
                        cache_mib=64, progress=None):
    """Index one scan, retaining *all* duplicate locations and blocking core conflicts.

    A prefix run is always audit-only. Never overwrites an existing index. SQLite
    bounds ID storage; the only Python key map contains the small profile roster.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch_size, cache_mib)):
        raise ValueError("batch_size and cache_mib must be positive integers")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    output = Path(output).resolve()
    users, manifest = load_users(output, require_ready=False)
    if manifest["status"] != "comments_pending":
        raise DataContractError("Bundle already indexed; choose a new output directory")
    for source in manifest["sources"].values():
        check_source(source)
    source_path = Path(source_path).resolve()
    database = output / "comments.sqlite"
    # Conservative reserve: three source sizes plus 2 GiB for full runs; a pilot
    # reserves only bounded pilot space. This is an estimate, not a disk quota.
    reserve = (source_path.stat().st_size * 3 if limit is None else
               min(source_path.stat().st_size * 3, limit * 2048)) + 2 * 1024**3
    free_before = shutil.disk_usage(output).free
    if free_before < reserve:
        raise OSError("Insufficient free disk for conservative index reserve")
    with database.open("xb"):
        pass
    start = time.perf_counter()
    db = sqlite3.connect(database)
    report = {"status": "indexing", "policy": POLICY, "core_fields": CORE_FIELDS,
              "journal_mode": "WAL",
              "limit": limit, "batch_size": batch_size, "cache_mib": cache_mib,
              "free_disk_before": free_before, "disk_reserve_estimate": reserve,
              "max_record_bytes": MAX_RECORD_BYTES, "python": sys.version,
              "sqlite": sqlite3.sqlite_version, "users_sha256": manifest["users_sha256"],
              "split_seed": manifest["seed"]}
    counts = Counter(records=0, unique_comments=0, duplicate_records=0,
                     exact_duplicates=0, metadata_variants=0, core_conflicts=0,
                     invalid_records=0, unlinked_records=0, normalized_id_records=0)
    author_map = {}  # Bounded by profile count; unknown authors are resolved in SQLite.

    def checkpoint():
        if counts["records"] % batch_size == 0:
            db.commit()
            if progress:
                progress({"records": counts["records"], "seconds": round(time.perf_counter()-start, 2),
                          "duplicates": counts["duplicate_records"], "core_conflicts": counts["core_conflicts"],
                          "peak_memory_bytes": peak_memory_bytes(), "database_bytes": database.stat().st_size})
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA wal_autocheckpoint=1000")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA temp_store=FILE")
        db.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
        db.execute("PRAGMA mmap_size=0")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        for user in users:
            if user.dataset_id == "pandora":
                cursor = db.execute("INSERT INTO authors(author,user_key,profile_exists,label_eligible,partition) "
                                    "VALUES (?,?,1,?,?)", (user.source_author, user.user_key,
                                                          int(user.label_eligible), user.partition))
                author_map[user.source_author] = cursor.lastrowid
        _put_metadata(db, report)
        conflict_fields = Counter()
        with CsvSource(source_path, REQUIRED_FIELDS) as source, source_path.open("rb") as reference:
            report["header"] = source.header
            insert = ("INSERT OR IGNORE INTO comments(comment_id,author_id,source_record,byte_start,byte_end,"
                      "line_start,line_end,core_sha,fields_sha) VALUES (?,?,?,?,?,?,?,?,?)")
            for row, span in source.rows(limit):
                counts["records"] += 1
                try:
                    comment_id = normalize_comment_id(row["id"])
                    if not row["author"] or row["author"] != row["author"].strip():
                        raise ValueError("author")
                    core = _core(row)
                except ValueError:
                    db.execute("INSERT INTO invalid_records VALUES (?,?,?,?,?,?,?)",
                               (span.record, span.byte_start, span.byte_end, span.line_start,
                                span.line_end, span.fields_sha256, "invalid_id_author_or_timestamp"))
                    counts["invalid_records"] += 1
                    checkpoint()
                    continue
                if row["id"] != _base36(comment_id):
                    counts["normalized_id_records"] += 1
                author_id = author_map.get(row["author"])
                if author_id is None:
                    found = db.execute("SELECT author_id FROM authors WHERE author=?", (row["author"],)).fetchone()
                    if found is None:
                        author_id = db.execute(
                            "INSERT INTO authors(author,user_key,profile_exists,label_eligible) VALUES (?,?,0,0)",
                            (row["author"], pandora_user_key(row["author"]))).lastrowid
                    else:
                        author_id = found[0]
                    counts["unlinked_records"] += 1
                core_sha, row_sha = fields_digest(core), bytes.fromhex(span.fields_sha256)
                cursor = db.execute(insert, (comment_id, author_id, span.record, span.byte_start,
                                             span.byte_end, span.line_start, span.line_end, core_sha, row_sha))
                if cursor.rowcount:
                    counts["unique_comments"] += 1
                else:
                    counts["duplicate_records"] += 1
                    previous = db.execute("SELECT source_record,byte_start,byte_end,line_start,line_end,"
                                          "fields_sha,core_sha FROM comments WHERE comment_id=?",
                                          (comment_id,)).fetchone()
                    seen = row_sha == previous[5] or db.execute(
                        "SELECT 1 FROM variants WHERE comment_id=? AND fields_sha=?",
                        (comment_id, row_sha)).fetchone() is not None
                    changed = []
                    if core_sha != previous[6]:
                        original = read_span(reference, source.header, _source_span(previous))
                        changed = [name for name, a, b in zip(CORE_FIELDS, _core(original), core) if a != b]
                        conflict_fields.update(changed)
                        kind = "core_conflict"
                        counts["core_conflicts"] += 1
                    elif seen:
                        kind = "exact"
                        counts["exact_duplicates"] += 1
                    else:
                        kind = "metadata_only"
                        counts["metadata_variants"] += 1
                    if not seen:
                        db.execute("INSERT INTO variants VALUES (?,?,?)", (comment_id, row_sha, span.record))
                    db.execute("INSERT INTO duplicates VALUES (?,?,?,?,?,?,?,?,?,?)",
                               (span.record, comment_id, span.byte_start, span.byte_end, span.line_start,
                                span.line_end, row_sha, kind, int(seen), json.dumps(changed)))
                    db.execute("UPDATE comments SET duplicate_count=duplicate_count+1,"
                               "conflict_count=conflict_count+? WHERE comment_id=?",
                               (int(bool(changed)), comment_id))
                checkpoint()
            source_info = source.fingerprint()
        db.commit()
        # Build the user retrieval index only after import; no in-memory ID set.
        db.execute("CREATE INDEX comments_author ON comments(author_id,source_record)")
        db.execute("CREATE INDEX duplicates_comment ON duplicates(comment_id,source_record)")
        summary = db.execute("SELECT COUNT(*),COALESCE(SUM(duplicate_count>0),0),"
                             "COALESCE(SUM(conflict_count>0),0),COALESCE(SUM(duplicate_count),0) "
                             "FROM comments").fetchone()
        if summary[0] != counts["unique_comments"] or summary[3] != counts["duplicate_records"]:
            raise DataContractError("Index accounting mismatch")
        linked = db.execute("SELECT COUNT(DISTINCT c.author_id) FROM comments c JOIN authors a "
                            "USING(author_id) WHERE a.profile_exists=1").fetchone()[0]
        supervised = db.execute("SELECT COUNT(*),COUNT(DISTINCT c.author_id) FROM comments c JOIN authors a "
                                "USING(author_id) WHERE a.label_eligible=1").fetchone()
        for user_source in manifest["sources"].values():
            check_source(user_source)
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise DataContractError("SQLite quick_check failed")
        report.update(counts)
        report.update(source=source_info, duplicated_ids=summary[1], conflicted_ids=summary[2],
                      conflict_field_counts=dict(conflict_fields), linked_profile_users=linked,
                      supervised_logical_comments=supervised[0], supervised_users_with_comments=supervised[1],
                      unknown_authors=db.execute("SELECT COUNT(*) FROM authors WHERE profile_exists=0").fetchone()[0],
                      elapsed_seconds=time.perf_counter()-start, peak_memory_bytes=peak_memory_bytes(),
                      free_disk_after=shutil.disk_usage(output).free)
        report["status"] = ("review_required" if counts["core_conflicts"] or counts["invalid_records"] else
                            "ready" if source_info["complete"] and limit is None else "partial_audit")
        report["database_bytes"] = database.stat().st_size
        report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), Path(__file__).with_name("raw_data.py"))}
        _put_metadata(db, report)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        write_json(output / "comment_audit.json", report)
        manifest.update(status=report["status"], comment_source=source_info,
                        comment_index="comments.sqlite", comment_policy=POLICY,
                        comment_audit_sha256=hashlib.sha256((output / "comment_audit.json").read_bytes()).hexdigest())
        write_json(output / "manifest.json", manifest)
        return report
    except BaseException:
        db.rollback()
        manifest["status"] = "index_failed"
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        db.close()


def _base36(number):
    if number == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while number:
        number, remainder = divmod(number, 36)
        result = digits[remainder] + result
    return result


@dataclass(frozen=True)
class RawComment:
    comment_id: str
    user_key: str
    partition: str | None
    created_utc: str
    source_span: SourceSpan
    body: str = field(repr=False)


class CommentIndex:
    """Read-only source-backed access. No training access to pending/partial runs."""

    def __init__(self, output):
        output = Path(output).resolve()
        _, self.manifest = load_users(output)
        # load_users authenticates the complete overlay before accepting readiness.
        self.resolved = "resolution" in self.manifest
        base = (Path(self.manifest["resolution"]["base_directory"]).resolve()
                if self.resolved else output)
        check_sqlite_snapshot(base / "comments.sqlite")
        audit = (base / "comment_audit.json").read_bytes()
        if hashlib.sha256(audit).hexdigest() != self.manifest["comment_audit_sha256"]:
            raise DataContractError("Comment audit digest mismatch")
        self.report = json.loads(audit)
        if self.report["policy"] != POLICY:
            raise DataContractError("Unsupported comment identity policy")
        if (self.report["users_sha256"], self.report["split_seed"]) != (
                self.manifest["users_sha256"], self.manifest["seed"]):
            raise DataContractError("Comment index belongs to a different user manifest")
        self.path = check_source(self.report["source"])
        self.db = sqlite3.connect((base / "comments.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            stored = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
            allowed = ("ready", "review_required") if self.resolved else ("ready",)
            if stored != self.report or stored["status"] not in allowed:
                raise ReviewRequiredError("Comment index is not ready or metadata differs")
            if self.resolved:
                self.db.execute("ATTACH DATABASE ? AS resolution",
                                ((output / "resolution.sqlite").as_uri()+"?mode=ro&immutable=1",))
                policy = self.db.execute("SELECT value FROM resolution.metadata WHERE key='policy'").fetchone()
                if policy != (self.manifest["resolution"]["policy"],):
                    raise DataContractError("Resolution database policy mismatch")
            self.db.execute("PRAGMA query_only=ON")
        except BaseException:
            self.db.close()
            raise

    def iter_user_comments(self, user_key):
        """Yield canonical raw comments in *source order*, not chronological order."""
        user = self.db.execute("SELECT author_id,partition FROM authors WHERE user_key=? AND profile_exists=1",
                               (user_key,)).fetchone()
        if user is None:
            raise KeyError("Unknown profile user key")
        check_source(self.report["source"])
        exclusion = (" AND NOT EXISTS (SELECT 1 FROM resolution.decisions r "
                     "WHERE r.comment_id=c.comment_id AND r.action='quarantine')" if self.resolved else "")
        rows = self.db.execute("SELECT source_record,byte_start,byte_end,line_start,line_end,fields_sha,"
                               "comment_id FROM comments c WHERE author_id=?" + exclusion +
                               " ORDER BY source_record", (user[0],))
        with self.path.open("rb") as source:
            for record in rows:
                span = _source_span(record)
                raw = read_span(source, self.report["header"], span)
                try:
                    source_id = normalize_comment_id(raw["id"])
                except ValueError as exc:
                    raise DataContractError("Invalid source comment ID") from exc
                if source_id != record[6]:
                    raise DataContractError("Indexed comment ID differs from its source record")
                if pandora_user_key(raw["author"]) != user_key:
                    raise DataContractError("Indexed comment author association mismatch")
                yield RawComment(_base36(record[6]), user_key, user[1], normalize_timestamp(raw["created_utc"]),
                                 span, raw["body"])

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def iter_conflict_locations(output):
    """Explicit audit access to *both* source versions; never chooses a winner."""
    output = Path(output).resolve()
    _, manifest = load_users(output, require_ready=False)
    audit = (output / "comment_audit.json").read_bytes()
    if hashlib.sha256(audit).hexdigest() != manifest["comment_audit_sha256"]:
        raise DataContractError("Comment audit digest mismatch")
    report = json.loads(audit)
    check_source(report["source"])
    db = sqlite3.connect((output / "comments.sqlite").as_uri() + "?mode=ro", uri=True)
    try:
        stored = json.loads(db.execute("SELECT value FROM metadata WHERE key='report'").fetchone()[0])
        if stored != report or report["users_sha256"] != manifest["users_sha256"]:
            raise DataContractError("Audit index and manifest differ")
        rows = db.execute(
            "SELECT c.comment_id,c.source_record,c.byte_start,c.byte_end,c.line_start,c.line_end,c.fields_sha,"
            "d.source_record,d.byte_start,d.byte_end,d.line_start,d.line_end,d.fields_sha,d.core_fields "
            "FROM duplicates d JOIN comments c USING(comment_id) "
            "WHERE d.kind='core_conflict' ORDER BY d.source_record")
        for row in rows:
            yield {"comment_id": _base36(row[0]), "canonical": _source_span(row[1:7]),
                   "variant": _source_span(row[7:13]), "changed_fields": tuple(json.loads(row[13]))}
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Index the first N comments as a partial audit")
    parser.add_argument("--cache-mib", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=20000)
    args = parser.parse_args()
    prepare_users(args.root, args.output, seed=args.seed)
    previous = [0.0]

    def progress(info):
        if info["seconds"] - previous[0] >= 30:
            print(json.dumps(info), flush=True)
            previous[0] = info["seconds"]

    report = build_comment_index(args.root / "pandora/all_comments_since_2015.csv", args.output,
                                 limit=args.limit, cache_mib=args.cache_mib,
                                 batch_size=args.batch_size, progress=progress)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("source", "header", "python", "code_sha256")}, sort_keys=True), flush=True)
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
