"""Versioned raw-user records, label eligibility and user-level splits.

Manifests contain private source metadata. Keep them in a local artifact directory.
"""

import csv
import hashlib
import io
import json
import sqlite3
import struct
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .data import DataContractError
from .labels import MBTI_TYPES, mbti_to_bits, mbti_to_class_id

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 16 * 1024 * 1024
SPLIT_ALGORITHM = "sha256-seeded-user-key-v1; floor(3N/5),floor(N/5),remainder"


class ReviewRequiredError(DataContractError):
    """An audit is incomplete or contains unresolved source conflicts."""


def fields_digest(values):
    """Length-framed UTF-8 field digest, independent of CSV quoting/newlines."""
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.digest()


@dataclass(frozen=True)
class SourceSpan:
    record: int  # One-based logical data record, excluding header.
    byte_start: int
    byte_end: int
    line_start: int
    line_end: int
    fields_sha256: str


class CsvSource:
    """Strict UTF-8 CSV stream with exact byte spans and a one-pass file hash.

    At most one logical record is parsed at a time, capped at max_record_bytes.
    CSV's process-global field limit is restored on exit; not a threaded reader.
    """

    def __init__(self, path, required, *, max_record_bytes=MAX_RECORD_BYTES):
        self.path = Path(path).resolve()
        self.required = tuple(required)
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int) or max_record_bytes < 1:
            raise ValueError("max_record_bytes must be a positive integer")
        self.limit = max_record_bytes
        self.offset = self.line = self.record = 0
        self.start = 0
        self.digest = hashlib.sha256()

    def __enter__(self):
        self.before = self.path.stat()
        self.stream = self.path.open("rb")
        self.old_limit = csv.field_size_limit()
        csv.field_size_limit(self.limit)
        self.reader = csv.reader(self, strict=True)
        try:
            try:
                self.header = next(self.reader)
            except (StopIteration, csv.Error) as exc:
                raise DataContractError(f"{self.path.name}: missing or malformed CSV header") from exc
            if len(set(self.header)) != len(self.header) or not set(self.required) <= set(self.header):
                raise DataContractError(f"{self.path.name}: missing/duplicate CSV columns")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.stream.readline(max(1, self.limit - (self.offset - self.start) + 1))
        if not raw:
            raise StopIteration
        self.offset += len(raw)
        if self.offset - self.start > self.limit:
            raise DataContractError(f"{self.path.name}: record exceeds {self.limit} bytes")
        self.digest.update(raw)
        self.line += 1
        try:
            return raw.decode("utf-8-sig" if self.line == 1 else "utf-8")
        except UnicodeError as exc:
            raise DataContractError(f"{self.path.name}: invalid UTF-8 at physical line {self.line}") from exc

    def rows(self, limit=None):
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("record limit must be a positive integer")
        while limit is None or self.record < limit:
            self.start, line_start = self.offset, self.line + 1
            try:
                values = next(self.reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise DataContractError(f"{self.path.name}: malformed CSV near physical line {self.line}") from exc
            self.record += 1
            if len(values) != len(self.header):
                raise DataContractError(f"{self.path.name}: record {self.record}: wrong column count")
            span = SourceSpan(self.record, self.start, self.offset, line_start, self.line,
                              fields_digest(values).hex())
            yield dict(zip(self.header, values)), span

    def fingerprint(self):
        after = self.path.stat()
        if (after.st_size, after.st_mtime_ns) != (self.before.st_size, self.before.st_mtime_ns):
            raise DataContractError(f"{self.path.name}: source changed during reading")
        complete = self.offset == after.st_size
        return {"path": str(self.path), "size": after.st_size, "mtime_ns": after.st_mtime_ns,
                "header": self.header, "records_scanned": self.record,
                "bytes_scanned": self.offset, "complete": complete,
                "sha256": self.digest.hexdigest() if complete else None,
                "scanned_prefix_sha256": self.digest.hexdigest()}

    def __exit__(self, *args):
        self.stream.close()
        csv.field_size_limit(self.old_limit)


def _sqlite_snapshot_state(path):
    path = Path(path)
    wal = Path(str(path)+"-wal")
    if wal.exists() and wal.stat().st_size:
        raise DataContractError("Nonempty SQLite WAL: close writers and checkpoint with SQLite upstream; a main-file snapshot cannot include WAL contents")
    journal = Path(str(path)+"-journal")
    if journal.exists() and journal.stat().st_size:
        size = journal.stat().st_size
        with journal.open("rb") as stream:
            header = stream.read(28)
        if size >= 512 and header == bytes(28):
            return "zero_header_journal"
        raise DataContractError("Active/hot or unknown rollback journal: close writers and recover with SQLite upstream before preparing a snapshot")
    return "clear"


def check_sqlite_snapshot(path):
    """Require a complete immutable main-file snapshot, without modifying it.

    A zeroed PERSIST journal is not a hot journal. It still requires explicit
    preparation into a new snapshot: a header alone cannot establish quiescence
    for an immutable reader. Empty WAL/SHM files remain harmless bookkeeping.
    """
    path = Path(path)
    if _sqlite_snapshot_state(path) == "zero_header_journal":
        raise DataContractError("Zero-header rollback journal (possibly inactive PERSIST): use prepare_sqlite_snapshot(source, new_destination) before recording the new snapshot identity; the source is left unchanged")
    return path


def prepare_sqlite_snapshot(source, destination, *, timeout=30.0):
    """Explicitly back up committed SQLite data into a NEW, clean main file.

    Accepts clear or inactive-PERSIST sources, never nonempty WAL or ambiguous
    rollback journals. SQLite's read-only backup API supplies a consistent view;
    it does not trust zeroed headers as proof that no writer exists. The source
    is not recovered, checkpointed, reconfigured or overwritten. The destination
    must be fingerprinted/approved by its data producer, not substituted into an
    existing accepted bundle. A failed backup can leave its new partial file.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be a finite positive number of seconds")
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    if not source.is_file():
        raise DataContractError("SQLite source must be a regular file")
    _sqlite_snapshot_state(source)
    # Reserve a new path before SQLite opens it; no existing destination is lost.
    with destination.open("xb"):
        pass
    deadline = time.monotonic()+timeout

    def progress(status, remaining, total):
        if time.monotonic() > deadline:
            raise DataContractError("SQLite backup timed out; close writers upstream and use a new destination")

    try:
        with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True, timeout=timeout)) as reader, \
                closing(sqlite3.connect(destination, timeout=timeout)) as writer:
            reader.backup(writer, pages=256, progress=progress, sleep=0.01)
            writer.execute("PRAGMA journal_mode=DELETE")
            if writer.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise DataContractError("Prepared SQLite snapshot failed integrity checking")
        _sqlite_snapshot_state(source)
        check_sqlite_snapshot(destination)
    except sqlite3.Error as exc:
        raise DataContractError("Cannot prepare SQLite snapshot; source may need upstream recovery; a partial new destination may remain") from exc
    return destination


def check_source(source):
    path = Path(source["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
        raise DataContractError(f"{path.name}: source version changed")
    return path


def read_span(stream, header, span):
    """Read and verify one source location, without disclosing its fields in errors."""
    length = span.byte_end - span.byte_start
    if length < 1 or length > MAX_RECORD_BYTES:
        raise DataContractError("Invalid or oversized source span")
    stream.seek(span.byte_start)
    raw = stream.read(length)
    previous = csv.field_size_limit()
    try:
        csv.field_size_limit(MAX_RECORD_BYTES)
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise DataContractError(f"Cannot parse source record {span.record}") from exc
    finally:
        csv.field_size_limit(previous)
    if len(rows) != 1 or len(rows[0]) != len(header) or fields_digest(rows[0]).hex() != span.fields_sha256:
        raise DataContractError(f"Source record {span.record}: fields digest mismatch")
    return dict(zip(header, rows[0]))


@dataclass(frozen=True)
class UserRecord:
    dataset_id: str
    user_key: str
    source_span: SourceSpan
    mbti: str | None
    bits: tuple[int, int, int, int] | None
    class_id: int | None
    label_eligible: bool
    label_status: str
    partition: str | None = None
    sample_eligible: bool | None = None  # Assigned by text processing.
    source_author: str | None = field(default=None, repr=False)
    raw_profile: dict | None = field(default=None, repr=False)


def pandora_user_key(author):
    if not isinstance(author, str) or not author or author != author.strip():
        raise DataContractError("Missing or whitespace-padded Pandora author")
    return "pandora:" + hashlib.sha256(author.encode("utf-8")).hexdigest()


def _label(raw):
    value = raw.strip().upper()
    if value in MBTI_TYPES:
        return value, mbti_to_bits(value), mbti_to_class_id(value), True, "complete"
    if not value:
        status = "missing"
    elif len(value) == 4 and "X" in value and all(
        letter in (pair + "X") for letter, pair in zip(value, ("IE", "NS", "TF", "PJ"))
    ):
        status = "incomplete"
    else:
        status = "invalid"
    return None, None, None, False, status


def read_raw_users(root):
    """Read user metadata only; Kaggle posts are not retained in memory."""
    root = Path(root)
    users, sources = [], {}
    kaggle_rows = []
    with CsvSource(root / "kaggle/mbti_1.csv", ("type", "posts")) as source:
        if source.header != ["type", "posts"]:
            raise DataContractError("Kaggle requires the original type,posts schema")
        for row, span in source.rows():
            kaggle_rows.append((span, _label(row["type"])))
        sources["kaggle"] = source.fingerprint()
    version = sources["kaggle"]["sha256"]
    for span, label in kaggle_rows:
        users.append(UserRecord("kaggle", f"kaggle:{version}:{span.record}", span, *label))
    seen = set()
    with CsvSource(root / "pandora/author_profiles.csv", ("author", "mbti")) as source:
        for row, span in source.rows():
            author = row["author"]
            key = pandora_user_key(author)
            if key in seen:
                raise DataContractError(f"Duplicate profile author at record {span.record}")
            seen.add(key)
            users.append(UserRecord("pandora", key, span, *_label(row["mbti"]),
                                    source_author=author, raw_profile=row))
        sources["pandora"] = source.fingerprint()
    return users, sources


def split_users(users, *, seed=0):
    """Stable hash ranking, independent of input ordering and personality labels."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    users = list(users)
    if len({u.user_key for u in users}) != len(users):
        raise DataContractError("Duplicate user keys")
    for user in users:
        if user.label_eligible and (user.bits != mbti_to_bits(user.mbti)
                                    or user.class_id != mbti_to_class_id(user.mbti)):
            raise DataContractError("User label encoding is inconsistent")
    assignment = {}
    for dataset in sorted({u.dataset_id for u in users}):
        eligible = [u for u in users if u.dataset_id == dataset and u.label_eligible]
        eligible.sort(key=lambda u: (hashlib.sha256(
            json.dumps(["emoperso-split-v1", seed, dataset, u.user_key],
                       separators=(",", ":")).encode("utf-8")).digest(), u.user_key))
        train, validation = len(eligible) * 3 // 5, len(eligible) // 5
        for i, user in enumerate(eligible):
            assignment[user.user_key] = ("train" if i < train else "validation"
                                        if i < train + validation else "test")
    return [replace(u, partition=assignment.get(u.user_key)) for u in users]


def user_counts(users):
    result = {}
    for user in users:
        counts = result.setdefault(user.dataset_id, Counter())
        counts["total"] += 1
        counts[user.label_status] += 1
        if user.partition:
            counts[user.partition] += 1
    return {name: dict(counts) for name, counts in result.items()}


def write_json(path, value):
    """Replace only a derived metadata file owned by this run."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    temporary.replace(path)


def prepare_users(root, output, *, seed=0):
    """Create a new private bundle. Existing output directories are never reused."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new run directory")
    users, sources = read_raw_users(root)
    users = split_users(users, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with (output / "users.jsonl").open("xb") as stream:
        for user in sorted(users, key=lambda u: u.user_key):
            raw = (json.dumps(asdict(user), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(raw)
            digest.update(raw)
    metadata = {"schema_version": SCHEMA_VERSION, "status": "comments_pending",
                "seed": seed, "split_algorithm": SPLIT_ALGORITHM, "sources": sources,
                "user_counts": user_counts(users), "users_sha256": digest.hexdigest(),
                "sample_eligibility": "not_assessed", "privacy": "local_private_artifact"}
    write_json(output / "manifest.json", metadata)
    return metadata


def load_users(output, *, require_ready=True):
    """Load a bundle with verified user digests, labels and resolution decisions.

    Audit callers may inspect pending bundles explicitly. Source-backed readers
    require a ready bundle and authenticated resolution decisions.
    """
    output = Path(output)
    metadata = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError("Unsupported raw-data schema version")
    if require_ready and metadata["status"] != "ready":
        raise ReviewRequiredError(f"Raw bundle is not ready: {metadata['status']}")
    if require_ready and "resolution" in metadata:
        from .conflict_resolution import validate_resolution
        validate_resolution(output, metadata)
    users, digest = [], hashlib.sha256()
    with (output / "users.jsonl").open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            value = json.loads(raw)
            value["source_span"] = SourceSpan(**value["source_span"])
            if value["bits"] is not None:
                value["bits"] = tuple(value["bits"])
            user = UserRecord(**value)
            if user.label_eligible:
                if (user.bits != mbti_to_bits(user.mbti) or user.class_id != mbti_to_class_id(user.mbti)
                        or user.partition not in ("train", "validation", "test")):
                    raise DataContractError("Inconsistent supervised user record")
            elif any(v is not None for v in (user.mbti, user.bits, user.class_id, user.partition)):
                raise DataContractError("Unlabelled user has supervised fields")
            users.append(user)
    if digest.hexdigest() != metadata["users_sha256"] or user_counts(users) != metadata["user_counts"]:
        raise DataContractError("User manifest integrity check failed")
    if len({u.user_key for u in users}) != len(users):
        raise DataContractError("Duplicate user keys in manifest")
    return users, metadata
