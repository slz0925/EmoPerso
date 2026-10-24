"""Public CSV/JSONL user datasets and bounded NumPy batch iteration.

All supplied files are validated before the first user is yielded. A private,
temporary SQLite index groups posts and checks identities on disk.
"""

import hashlib
import json
import random
import sqlite3
import tempfile
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from .comment_index import normalize_timestamp
from .errors import DataContractError
from .labels import mbti_to_bits
from .raw_data import CsvSource, SourceSpan, fields_digest
from .text_data import (CleanText, TextConfig, TextPost, UserText, clip_prefix,
                        guard_clipped_posts, join_posts, find_explicit_type,
                        clean_text, has_content, text_sha)
from .token_inputs import validate_identifier, post_key, user_text

_PARTITIONS = ("train", "validation", "test", "inference")


def _json_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DataContractError("Duplicate JSON field")
        obj[key] = value
    return obj


def _records(path, max_record_bytes):
    if path.suffix.lower() == ".csv":
        with CsvSource(path, ("user_id", "text"), max_record_bytes=max_record_bytes) as source:
            for row, span in source.rows():
                yield row, span
    elif path.suffix.lower() == ".jsonl":
        with path.open("rb") as stream:
            record = 0
            while True:
                start = stream.tell()
                raw = stream.readline(max_record_bytes+1)
                if not raw:
                    break
                record += 1
                if len(raw) > max_record_bytes:
                    raise DataContractError(f"{path.name}: record {record} exceeds byte limit")
                try:
                    row = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
                except (UnicodeError, ValueError) as exc:
                    raise DataContractError(f"{path.name}: invalid JSON at record {record}") from exc
                if not isinstance(row, dict):
                    raise DataContractError(f"{path.name}: record {record} must be a JSON object")
                canonical = json.dumps(row, sort_keys=True, ensure_ascii=True)
                span = SourceSpan(record, start, stream.tell(), record, record, fields_digest([canonical]).hex())
                yield row, span
    else:
        raise DataContractError("Dataset files must use .csv or .jsonl")


def _normalize(row, dataset_id):
    if not {"user_id", "text"} <= row.keys():
        raise DataContractError("Missing user_id or text")
    key, text = row["user_id"], row["text"]
    dataset = row.get("dataset_id")
    if dataset in (None, ""):dataset = dataset_id
    validate_identifier(key, "user_id")
    validate_identifier(dataset, "dataset_id")
    if not isinstance(text, str) or not text.strip():
        raise DataContractError("text must be a nonempty string")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataContractError("Malformed text Unicode") from exc
    labels = row.get("labels")
    columns = [row.get("label"+str(i)) for i in range(4)]
    present = [v not in (None, "") for v in columns]
    if any(present):
        if not all(present) or labels is not None:
            raise DataContractError("Supply all four label columns or labels, not both")
        labels = []
        for value in columns:
            if isinstance(value, str):
                if value.strip() not in ("0", "1"):
                    raise DataContractError("Label columns must contain binary values")
                value = int(value.strip())
            labels.append(value)
    if row.get("mbti") not in (None, ""):
        try:
            mbti_labels = mbti_to_bits(row["mbti"])
        except ValueError as exc:
            raise DataContractError("Invalid complete MBTI label") from exc
        if labels is not None and tuple(labels) != mbti_labels:
            raise DataContractError("MBTI string disagrees with binary labels")
        labels = mbti_labels
    part = row.get("partition")
    if part in (None, ""):part = "inference"
    if part not in _PARTITIONS:
        raise DataContractError("partition must be train, validation, test or inference")
    sample = user_text(key, text, labels=labels, dataset_id=dataset,
                       partition=None if part == "inference" else part)
    if sample.label_eligible != (part != "inference"):
        raise DataContractError("Supervised rows require labels and a split; inference rows must be unlabelled")
    return sample


class UserDataset:
    """Context-managed, validated disk index. Text memory is bounded per record/user.

    layout='users': exactly one row per (dataset_id,user_id).
    layout='posts': rows share that identity, but require distinct post_id and
    identical labels/partition. order='recent' requires decimal UTC timestamps;
    ties follow source order. No automatic split or label-based sampling occurs.
    """

    def __init__(self, path, *, partition=None, dataset_id="custom", layout="users",
                 preprocess=False, order="source", text_config=TextConfig(),
                 temp_dir=None, max_record_bytes=1024*1024):
        if partition not in (None, *_PARTITIONS) or layout not in ("users", "posts") or order not in ("source", "recent"):
            raise ValueError("Invalid partition, layout or post order")
        if not isinstance(preprocess, bool) or not isinstance(text_config, TextConfig):
            raise ValueError("Expected boolean preprocess and TextConfig")
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int) or max_record_bytes < 1:
            raise ValueError("max_record_bytes must be a positive integer")
        validate_identifier(dataset_id, "dataset_id")
        self.sources = tuple(Path(p).resolve() for p in ([path] if isinstance(path, (str, Path)) else path))
        if not self.sources or len(set(self.sources)) != len(self.sources):
            raise DataContractError("Provide distinct dataset files")
        self.partition, self.preprocess, self.order, self.layout = partition, preprocess, order, layout
        self.text_config, self.closed, self.db, self._temp = text_config, False, None, None
        self.source_versions = {}
        try:
            self._temp = tempfile.TemporaryDirectory(prefix="emoperso_dataset_", dir=temp_dir)
            self._owned_path = Path(self._temp.name).resolve()
            self._temp_parent = self._owned_path.parent
            self.db = sqlite3.connect(self._owned_path/"users.sqlite")
            self.db.executescript("""
                PRAGMA cache_size=-4096; PRAGMA temp_store=FILE;
                CREATE TABLE users (dataset TEXT, uid TEXT, partition TEXT, labels TEXT, ordinal INTEGER,
                                    PRIMARY KEY(dataset,uid));
                CREATE TABLE posts (dataset TEXT, uid TEXT, post_id TEXT, ordinal INTEGER, body TEXT,
                                    utc TEXT, file_id INTEGER, span TEXT,
                                    PRIMARY KEY(dataset,uid,post_id));
                CREATE INDEX post_users ON posts(dataset,uid,ordinal);
            """)
            self.db.create_collation("utc_decimal", lambda a,b:(Decimal(a)>Decimal(b))-(Decimal(a)<Decimal(b)))
            ordinal = 0
            for file_id, source in enumerate(self.sources):
                if not source.is_file():raise DataContractError(f"Missing dataset file {source.name}")
                stat = source.stat(); self.source_versions[str(source)] = (stat.st_size, stat.st_mtime_ns)
                try:
                    for row, span in _records(source, max_record_bytes):
                        ordinal += 1
                        try:
                            sample = _normalize(row, dataset_id)
                            part = sample.partition or "inference"
                            labels = json.dumps(sample.bits) if sample.bits is not None else ""
                            old = self.db.execute("SELECT partition,labels FROM users WHERE dataset=? AND uid=?",
                                                  (sample.dataset_id,sample.user_key)).fetchone()
                            if old:
                                if old != (part, labels):
                                    raise DataContractError("User repeats across partitions or has conflicting labels")
                                if layout == "users":
                                    raise DataContractError("Duplicate user record")
                            else:
                                self.db.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                                    (sample.dataset_id,sample.user_key,part,labels,ordinal))
                            post_id = row.get("post_id") if layout == "posts" else "user-text"
                            validate_identifier(post_id, "post_id")
                            utc = None
                            if order == "recent":
                                utc = normalize_timestamp(str(row.get("created_utc", "")))
                            self.db.execute("INSERT INTO posts VALUES (?,?,?,?,?,?,?,?)",
                                (sample.dataset_id,sample.user_key,post_id,ordinal,sample.text,utc,file_id,json.dumps(span.__dict__)))
                        except (ValueError, TypeError, sqlite3.IntegrityError) as exc:
                            message = str(exc) if isinstance(exc, DataContractError) else "Invalid schema, timestamp or duplicate post_id"
                            raise DataContractError(f"{source.name}: record {span.record}: {message}") from exc
                        if ordinal % 2000 == 0:
                            self.db.commit()
                    after = source.stat()
                    if self.source_versions[str(source)] != (after.st_size,after.st_mtime_ns):
                        raise DataContractError("Dataset changed while indexing")
                except (OSError, UnicodeError) as exc:
                    raise DataContractError(f"Cannot read dataset file {source.name}") from exc
            if not ordinal:
                raise DataContractError("Dataset contains no records")
            self.db.commit()
            self.counts = dict(self.db.execute("SELECT partition,count(*) FROM users GROUP BY partition"))
        except BaseException as exc:
            try:
                self.close()
            except OSError as cleanup_error:
                raise DataContractError("Temporary index failed and cleanup was denied; choose a writable temp_dir.") from exc
            if isinstance(exc,(OSError,sqlite3.OperationalError)):
                raise DataContractError("Cannot create/read temporary dataset index; check source files, free disk space and a writable temp_dir") from exc
            raise

    def _check(self):
        if self.closed:
            raise DataContractError("Dataset is closed")

    def __len__(self):
        self._check()
        return sum(self.counts.values()) if self.partition is None else self.counts.get(self.partition,0)

    def _sample(self, dataset, uid, part, labels):
        config = self.text_config
        ordering = "utc COLLATE utc_decimal DESC,ordinal" if self.order == "recent" else "ordinal"
        cursor = self.db.execute("SELECT post_id,body,utc,file_id,span FROM posts WHERE dataset=? AND uid=? ORDER BY "+ordering,(dataset,uid))
        posts, operations, invalid = [], Counter(), Counter()
        raw = valid = 0
        original_has_type = False
        for post_id, body, utc, file_id, span in cursor:
            raw += 1
            result = clean_text(body) if self.preprocess else CleanText(body,(),None if has_content(body) else "no_content")
            operations.update(dict(result.operations))
            if result.reason:
                invalid[result.reason] += 1; continue
            clipped = clip_prefix(result.text,config.max_post_chars if self.layout=="posts" else config.max_user_chars)
            if len(clipped)<len(result.text):operations["post_character_clips"] += 1
            if not has_content(clipped):
                invalid["no_content_after_post_character_limit"] += 1; continue
            valid += 1
            if len(posts) < config.max_posts:
                original_has_type |= bool(find_explicit_type(result.text))
                posts.append(TextPost(post_key(dataset,uid,post_id),SourceSpan(**json.loads(span)),raw-1,None,utc,
                    text_sha(body),len(body),len(result.text),result.operations,clipped,
                    original_post_id=post_id if self.layout=="posts" else None,source_file_id=file_id))
        selected, size = [], 0
        for post in posts:
            available = config.max_user_chars-size-(2 if selected else 0)
            if available<=0:break
            retained = clip_prefix(post.text,available)
            if len(retained)<len(post.text):operations["user_character_clips"] += 1
            if not has_content(retained):
                operations["user_character_budget_no_content_posts"] += 1; continue
            selected.append(replace(post,text=retained));size += len(retained)+(2 if len(selected)>1 else 0)
        selected, text = join_posts(selected)
        # With preprocess=False, retain existing labels for the encoder to reject.
        # Label-free input still needs a check at the final clipped boundaries.
        if self.preprocess or not original_has_type:
            selected, text = guard_clipped_posts(selected,operations)
        sample = user_text(uid,text,labels=json.loads(labels) if labels else None,
                           partition=None if part=="inference" else part,dataset_id=dataset)
        return replace(sample,posts=tuple(selected),policy="public-clean-posts-v2" if self.preprocess else "public-provided-posts-v2",
            raw_post_count=raw,nonempty_raw_post_count=raw,valid_post_count=valid,candidate_post_count=len(posts),
            operations=tuple(sorted(operations.items())),invalid_post_reasons=tuple(sorted(invalid.items())))

    def iter_users(self, *, shuffle=False, seed=None):
        self._check()
        effective = self.partition or (next(iter(self.counts)) if len(self.counts)==1 else None)
        if not isinstance(shuffle,bool) or shuffle and (effective!="train" or isinstance(seed,bool) or not isinstance(seed,int)):
            raise ValueError("Shuffle requires a train-only view and an explicit integer seed")
        where = " WHERE partition=?" if self.partition is not None else ""
        args = (self.partition,) if self.partition is not None else ()
        if shuffle:
            self.db.create_function("shuffle_key",2,lambda d,u:hashlib.sha256(json.dumps([seed,d,u],ensure_ascii=True).encode()).digest())
            ordering = "shuffle_key(dataset,uid),dataset,uid"
        else:
            ordering = "ordinal"
        for dataset,uid,part,labels in self.db.execute("SELECT dataset,uid,partition,labels FROM users"+where+" ORDER BY "+ordering,args):
            self._check()
            yield self._sample(dataset,uid,part,labels)

    def __iter__(self):
        return self.iter_users()

    def close(self):
        if self.db is not None:
            self.db.close(); self.db = None
        if self._temp is not None:
            target = Path(self._temp.name).resolve()
            if target != self._owned_path or target.parent != self._temp_parent:
                raise RuntimeError("Temporary dataset path changed; refusing cleanup")
            self._temp.cleanup(); self._temp = None
        self.closed = True

    def __enter__(self):
        self._check(); return self

    def __exit__(self,*args):
        self.close()


def load_dataset(path, **kwargs):
    """Validate all provided files and return a closeable UserDataset."""
    return UserDataset(path,**kwargs)


def iter_batches(dataset, *, tokenizer, batch_size=8, shuffle=False, seed=None,
                 require_supervised=True, pad_to_max_length=False, shuffle_buffer_size=128, temp_dir=None):
    """Real user-level batches, reusing the supplied encoder and collator.

    UserDataset supports full deterministic disk shuffle. Already prepared
    UserText iterables use a bounded shuffle buffer (train only, explicit seed).
    Such streams are checked incrementally; file datasets are checked completely
    before iteration. Invalid users raise instead of being silently skipped.
    """
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
        raise ValueError("batch_size must be a positive integer")
    if not all(isinstance(v,bool) for v in (shuffle,require_supervised,pad_to_max_length)):
        raise ValueError("Batch flags must be boolean")
    if isinstance(shuffle_buffer_size,bool) or not isinstance(shuffle_buffer_size,int) or shuffle_buffer_size<1:
        raise ValueError("shuffle_buffer_size must be a positive integer")
    if isinstance(dataset,UserDataset):
        parts = {dataset.partition} if dataset.partition is not None else set(dataset.counts)
        if len(parts)>1:
            raise DataContractError("Select one partition before batching a multi-partition dataset")
        source = dataset.iter_users(shuffle=shuffle,seed=seed)
    else:
        source = iter(dataset)
        if shuffle and (isinstance(seed,bool) or not isinstance(seed,int)):
            raise ValueError("Shuffle requires an explicit integer seed")
    rng, buffer, batch = random.Random(seed), [], []
    expected = None
    def checked():
        nonlocal expected
        # Only identity metadata is kept on disk; no unbounded Python ID set.
        with tempfile.TemporaryDirectory(prefix="emoperso_batch_",dir=temp_dir) as tmp:
            root=Path(tmp).resolve()
            db=sqlite3.connect(root/"seen.sqlite")
            try:
                db.execute("PRAGMA cache_size=-2048")
                db.execute("PRAGMA temp_store=FILE")
                db.execute("CREATE TABLE seen(dataset TEXT,uid TEXT,PRIMARY KEY(dataset,uid))")
                for sample in source:
                    if not isinstance(sample,UserText):raise DataContractError("Expected UserText samples")
                    tokenizer.validate_user(sample)
                    identity=(sample.partition,sample.label_eligible)
                    if expected is not None and expected!=identity:raise DataContractError("Dataset stream mixes partitions or label presence")
                    expected=identity
                    if shuffle and sample.partition!="train":raise DataContractError("Only training users can be shuffled")
                    try:db.execute("INSERT INTO seen VALUES (?,?)",(sample.dataset_id,sample.user_key))
                    except sqlite3.IntegrityError as exc:raise DataContractError("Duplicate user in batch stream") from exc
                    yield sample
            finally:
                db.close()
                if Path(tmp).resolve()!=root or not root.name.startswith("emoperso_batch_"):
                    raise RuntimeError("Temporary batch path changed; refusing cleanup")
    for sample in checked():
        if shuffle and not isinstance(dataset,UserDataset):
            if len(buffer)<shuffle_buffer_size:
                buffer.append(sample); continue
            i=rng.randrange(len(buffer));sample,buffer[i]=buffer[i],sample
        batch.append(tokenizer.encode_user(sample))
        if len(batch)==batch_size:
            yield tokenizer.batch(batch,require_supervised=require_supervised,pad_to_max_length=pad_to_max_length);batch=[]
    rng.shuffle(buffer)
    for sample in buffer:
        batch.append(tokenizer.encode_user(sample))
        if len(batch)==batch_size:
            yield tokenizer.batch(batch,require_supervised=require_supervised,pad_to_max_length=pad_to_max_length);batch=[]
    if batch:yield tokenizer.batch(batch,require_supervised=require_supervised,pad_to_max_length=pad_to_max_length)
