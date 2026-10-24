"""Bounded cleaning and selection of source-backed user text."""

import argparse
import hashlib
import heapq
import json
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path

from .comment_index import CommentIndex, normalize_comment_id, peak_memory_bytes
from .conflict_resolution import RESOLUTION_POLICY, file_sha
from .errors import DataContractError
from .raw_data import SourceSpan, check_source, load_users, read_span, write_json

TEXT_POLICY = "user-text-v4"
SEPARATOR = "\n\n"
_BAD_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad\u200b\u200e\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]")
_BAD_UNICODE = re.compile(r"[\ufffd\ud800-\udfff]")
_PLACEHOLDER = re.compile(r"\[(?:deleted|removed(?: by reddit)?)\]", re.I)
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"]+", re.I)
_GAP = r"(?:\s*[-_/·]\s*|\s*)"
_TYPE = re.compile(r"(?<![A-Za-z0-9_])[IE]"+_GAP+r"[NS]"+_GAP+r"[TF]"+_GAP+
                   r"[PJ](?:[-‑–—][AT])?s?(?![A-Za-z0-9_])", re.I)
_MARKER = re.compile(r"(?<![A-Za-z0-9_])(?:MBTI|Myers[- ]Briggs|16Personalities)(?![A-Za-z0-9_])", re.I)
_WIDE = re.compile(r"[\uff21-\uff3a\uff41-\uff5a]")
_WIDE_MAP = {c: c-0xfee0 for c in (*range(0xff21, 0xff3b), *range(0xff41, 0xff5b))}
_LEXICAL = re.compile(r"[^\W_]", re.UNICODE)
_NEEDS_WORK = re.compile("|".join((
    _BAD_CONTROL.pattern, _BAD_UNICODE.pattern, _PLACEHOLDER.pattern,
    _URL.pattern, _TYPE.pattern, _MARKER.pattern, _WIDE.pattern,
    r"[^\S \n]|\s{2}|^\s|\s$")), re.I)


def text_sha(text):
    # Hash malformed Unicode consistently before cleaning.
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def has_content(text):
    """Letters/numbers or Unicode other-symbols (including emoji), not punctuation alone."""
    return bool(_LEXICAL.search(text)) or any(unicodedata.category(c) == "So" for c in text)


def _word_character(char):
    return char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")


def clip_prefix(text, limit):
    """Bound a prefix without creating a new word by cutting inside a word.

    A word consists of Unicode letters/numbers, underscores and combining marks.
    If a cut bisects one, omit its entire partial prefix. Exact-length and shorter
    inputs are unchanged; punctuation/whitespace are not stripped or rewritten.
    An overlong first word may therefore leave an empty prefix, which the
    caller records as ineligible when no usable content remains.
    """
    end = min(len(text), limit)
    if 0 < end < len(text) and _word_character(text[end-1]) and _word_character(text[end]):
        while end > 0 and _word_character(text[end-1]):
            end -= 1
    return text[:end]


@dataclass(frozen=True)
class TextConfig:
    max_posts: int = 64
    max_post_chars: int = 4096
    max_user_chars: int = 65536
    policy: str = TEXT_POLICY

    def __post_init__(self):
        if self.policy != TEXT_POLICY:
            raise ValueError("Unsupported text policy")
        for value in (self.max_posts, self.max_post_chars, self.max_user_chars):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Text resource bounds must be positive integers")


@dataclass(frozen=True)
class CleanText:
    text: str = field(repr=False)
    operations: tuple[tuple[str, int], ...]
    reason: str | None


def clean_text(text):
    """Remove explicit personality labels, URLs and unusable text markers.

    Case, negation, words, punctuation, paragraph breaks, ZWJ/ZWNJ and emoji are
    retained. Only a matching shadow folds fullwidth Latin letters for type tags.
    """
    if not isinstance(text, str):
        raise DataContractError("Post body must be a string")
    if not text.strip():
        return CleanText("", (), "empty_or_whitespace")
    if not _NEEDS_WORK.search(text):
        reason = None if has_content(text) else "no_content_after_cleaning"
        return CleanText(text if reason is None else "", (), reason)
    operations = Counter()
    value, operations["invalid_unicode_characters"] = _BAD_UNICODE.subn(" ", text)
    value, operations["control_characters"] = _BAD_CONTROL.subn(" ", value)
    value, operations["deletion_placeholders"] = _PLACEHOLDER.subn(" ", value)

    def remove_url(match):
        url = match.group()
        stripped = url.rstrip(".,;:!?)]}")
        return " " + url[len(stripped):]

    value, operations["urls"] = _URL.subn(remove_url, value)
    value, operations["mbti_markers"] = _MARKER.subn(" ", value)
    # Removing a marker/type can expose another explicit type across its gap.
    # Iterate to a fixed point, with a finite cap and fail-closed diagnostic.
    for _ in range(8):
        shadow = value.translate(_WIDE_MAP) if _WIDE.search(value) else value
        pieces, start, count = [], 0, 0
        for match in _TYPE.finditer(shadow):
            pieces.extend((value[start:match.start()], " "))
            start, count = match.end(), count+1
        if not count:
            break
        pieces.append(value[start:])
        value = "".join(pieces)
        operations["explicit_mbti_types"] += count
    shadow = value.translate(_WIDE_MAP) if _WIDE.search(value) else value
    if _TYPE.search(shadow):
        operations["unresolved_explicit_label_posts"] += 1
        return CleanText("", tuple(sorted((k,n) for k,n in operations.items() if n)),
                         "unresolved_explicit_label_after_redaction")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    operations["whitespace_changed_posts"] = int(normalized != value)
    reason = None if has_content(normalized) else "no_content_after_cleaning"
    return CleanText(normalized if reason is None else "",
                     tuple(sorted((key, n) for key, n in operations.items() if n)), reason)


@dataclass(frozen=True)
class TextPost:
    post_key: str
    source_span: SourceSpan
    fragment_index: int | None
    comment_id: str | None
    created_utc: str | None
    raw_text_sha256: str
    raw_characters: int
    clean_characters: int
    operations: tuple[tuple[str, int], ...]
    text: str = field(repr=False)
    user_text_start: int = 0
    user_text_end: int = 0
    original_post_id: str | None = None
    source_file_id: int | None = None


@dataclass(frozen=True)
class UserText:
    dataset_id: str
    user_key: str
    partition: str | None
    mbti: str | None
    bits: tuple[int, int, int, int] | None
    label_eligible: bool
    policy: str
    posts: tuple[TextPost, ...]
    text: str = field(repr=False)
    raw_post_count: int
    nonempty_raw_post_count: int
    valid_post_count: int
    candidate_post_count: int
    operations: tuple[tuple[str, int], ...]
    invalid_post_reasons: tuple[tuple[str, int], ...]
    text_eligible: bool
    sample_eligible: bool  # Supervised TEXT eligibility, not formal-token eligibility.
    exclusion_reasons: tuple[str, ...]
    token_eligible: None = None

    def provenance(self):
        """Private metadata only; never copy raw or cleaned bodies into a manifest."""
        value = {key: getattr(self, key) for key in self.__dataclass_fields__ if key not in ("posts", "text")}
        value.update(text_sha256=text_sha(self.text), text_characters=len(self.text), selected_posts=[])
        for post in self.posts:
            row = {key: getattr(post, key) for key in post.__dataclass_fields__
                   if key != "text" and not (key in ("original_post_id", "source_file_id")
                                              and getattr(post, key) is None)}
            row["source_span"] = asdict(post.source_span)
            row.update(text_sha256=text_sha(post.text), retained_characters=len(post.text))
            value["selected_posts"].append(row)
        return value


def _fragments(value):
    """Yield even empty trailing fragments without allocating a list of all posts."""
    start = index = 0
    while True:
        end = value.find("|||", start)
        if end < 0:
            yield index, value[start:]
            break
        yield index, value[start:end]
        start, index = end+3, index+1


def join_posts(posts):
    parts, positioned, size = [], [], 0
    for post in posts:
        separator = SEPARATOR if positioned else ""
        begin = size+len(separator)
        positioned.append(replace(post, user_text_start=begin, user_text_end=begin+len(post.text)))
        parts.extend((separator, post.text))
        size = begin+len(post.text)
    return positioned, "".join(parts)


def find_explicit_type(text, posts=()):
    """Return the first (start, end) in one real post, never across post boundaries.

    Offsets refer to the joined text. For an unstructured UserText (no posts),
    the whole text is one segment. Passing a shorter prefix also checks newly
    exposed types at its cut. Fullwidth folding preserves character offsets.
    Callers must validate that posts cover the text with SEPARATOR-only gaps.
    """
    segments = ((p.user_text_start, p.user_text_end) for p in posts) if posts else ((0, len(text)),)
    for start, end in segments:
        if start >= len(text):
            break
        match = _TYPE.search(text[start:min(end, len(text))].translate(_WIDE_MAP))
        if match:
            return start+match.start(), start+match.end()
    return None


def guard_clipped_posts(posts, operations):
    """Guard character-cut prefixes within posts; separators are hard boundaries.

    Cross-post letter sequences are not explicit types in either source post.
    Preserve them without inserting marker text or rerunning the cleaner. Only
    clipping can expose a new within-post type in already cleaned source text.
    """
    posts, text = join_posts(posts)
    if not any(operations[key] for key in ("post_character_clips", "user_character_clips",
                                           "user_character_budget_no_content_posts")):
        return posts, text
    while True:
        match = find_explicit_type(text, posts)
        if match is None:
            return posts, text
        index = next(i for i, p in enumerate(posts) if p.user_text_end > match[0])
        post = posts[index]
        retained = clip_prefix(post.text, max(0, match[0]-post.user_text_start))
        operations["clipped_join_type_rollbacks"] += 1
        if has_content(retained):
            posts[index] = replace(post, text=retained)
        else:
            posts.pop(index)
            operations["clipped_join_no_content_posts"] += 1
        posts, text = join_posts(posts)


def organize_user(user, raw_posts, config=TextConfig()):
    """Consume source posts once, keeping at most max_posts bounded text candidates.

    raw_posts yields (raw_body, post_key, span, fragment_index, comment_id, UTC).
    Pandora chooses greatest timestamp, then smallest numeric comment ID, then
    smallest source record. Final order is recent-first so later token prefix
    truncation preserves the priority. Kaggle keeps valid fragments in source order.
    """
    if user.dataset_id not in ("kaggle", "pandora"):
        raise DataContractError("Unsupported user dataset")
    candidates, operations, invalid = [], Counter(), Counter()
    raw_count = nonempty = valid = 0
    for body, key, span, fragment, comment, utc in raw_posts:
        raw_count += 1
        nonempty += bool(body.strip())
        result = clean_text(body)
        operations.update(dict(result.operations))
        if result.reason:
            invalid[result.reason] += 1
            continue
        clipped = clip_prefix(result.text, config.max_post_chars)
        if len(clipped) < min(len(result.text), config.max_post_chars):
            operations["post_word_boundary_rollbacks"] += 1
        if len(clipped) < len(result.text):
            operations["post_character_clips"] += 1
        if not has_content(clipped):
            invalid["no_content_after_post_character_limit"] += 1
            continue
        valid += 1
        if user.dataset_id == "kaggle" and len(candidates) >= config.max_posts:
            continue
        if user.dataset_id == "pandora":
            if utc is None or comment is None:
                raise DataContractError("Pandora requires real comment identity and numeric time")
            priority = (Decimal(utc), -normalize_comment_id(comment), -span.record)
            if len(candidates) >= config.max_posts and priority <= candidates[0][0]:
                continue
        else:
            priority = (-fragment,)
        post = TextPost(key, span, fragment, comment, utc, text_sha(body), len(body),
                        len(result.text), result.operations, clipped)
        if user.dataset_id == "pandora":
            if len(candidates) < config.max_posts:
                heapq.heappush(candidates, (priority, post))
            else:
                heapq.heapreplace(candidates, (priority, post))
        else:
            candidates.append((priority, post))
    ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
    selected, size = [], 0
    for _, post in ordered:
        separator = SEPARATOR if selected else ""
        available = config.max_user_chars-size-len(separator)
        if available <= 0:
            break
        retained = clip_prefix(post.text, available)
        if len(retained) < min(len(post.text), available):
            operations["user_word_boundary_rollbacks"] += 1
        if not has_content(retained):
            operations["user_character_budget_no_content_posts"] += 1
            continue
        if len(retained) < len(post.text):
            operations["user_character_clips"] += 1
        begin = size+len(separator)
        selected.append(replace(post, text=retained, user_text_start=begin, user_text_end=begin+len(retained)))
        size = begin+len(retained)
    selected, text = guard_clipped_posts(selected, operations)
    reasons = []
    if not user.label_eligible:
        reasons.append("no_complete_personality_label")
    if not raw_count:
        reasons.append("no_raw_posts")
    elif not valid:
        reasons.append("no_valid_posts_after_cleaning")
    elif not text:
        reasons.append("no_content_after_clipped_join_guard" if operations["clipped_join_type_rollbacks"]
                       else "no_content_within_user_character_budget")
    return UserText(user.dataset_id, user.user_key, user.partition, user.mbti, user.bits,
                    user.label_eligible, config.policy, tuple(selected), text, raw_count, nonempty,
                    valid, len(candidates), tuple(sorted(operations.items())), tuple(sorted(invalid.items())),
                    bool(text), bool(text) and user.label_eligible, tuple(reasons))


class UserTextSource:
    """Read authenticated raw bundles with bounded per-user text views."""

    def __init__(self, bundle, config=TextConfig(), *, progress=None):
        self.bundle = Path(bundle).resolve()
        self.users, self.manifest = load_users(self.bundle)
        if self.manifest.get("resolution", {}).get("policy") != RESOLUTION_POLICY:
            raise DataContractError("User text requires the approved resolved comment bundle")
        self.config, self.progress = config, progress
        self.lookup = {u.user_key: u for u in self.users}
        self.index = CommentIndex(self.bundle)
        self.kaggle = None
        self.comments_read = 0
        try:
            self.kaggle = check_source(self.manifest["sources"]["kaggle"]).open("rb")
        except BaseException:
            self.index.close()
            raise

    def _raw_posts(self, user):
        if user.dataset_id == "kaggle":
            source = self.manifest["sources"]["kaggle"]
            check_source(source)
            row = read_span(self.kaggle, source["header"], user.source_span)
            # No type/profile/flair field is ever part of model text.
            for index, body in _fragments(row["posts"]):
                yield body, f"{user.user_key}:fragment:{index}", user.source_span, index, None, None
        elif user.dataset_id == "pandora":
            for comment in self.index.iter_user_comments(user.user_key):
                if comment.user_key != user.user_key or comment.partition != user.partition:
                    raise DataContractError("Comment view crossed a user or partition boundary")
                self.comments_read += 1
                if self.progress and self.comments_read % 10000 == 0:
                    self.progress({"pandora_comments_read": self.comments_read})
                yield (comment.body, f"pandora:comment:{comment.comment_id}", comment.source_span,
                       None, comment.comment_id, comment.created_utc)
        else:
            raise DataContractError("Unsupported user dataset")

    def get_user(self, user_key):
        user = self.lookup[user_key]
        return organize_user(user, self._raw_posts(user), self.config)

    def iter_users(self, *, dataset_id=None, partition=None, limit_per_dataset=None):
        if dataset_id not in (None, "kaggle", "pandora") or partition not in (None, "train", "validation", "test"):
            raise ValueError("Invalid dataset or partition")
        if limit_per_dataset is not None and (isinstance(limit_per_dataset, bool)
                or not isinstance(limit_per_dataset, int) or limit_per_dataset < 1):
            raise ValueError("limit_per_dataset must be a positive integer")
        counts = Counter()
        for user in sorted(self.users, key=lambda u: (u.dataset_id, u.source_span.record)):
            if dataset_id is not None and user.dataset_id != dataset_id:
                continue
            if partition is not None and user.partition != partition:
                continue
            if limit_per_dataset is not None and counts[user.dataset_id] >= limit_per_dataset:
                continue
            counts[user.dataset_id] += 1
            yield self.get_user(user.user_key)

    def close(self):
        self.index.close()
        if self.kaggle:
            self.kaggle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TextStatistics:
    def __init__(self):
        self.groups = {}

    def update(self, sample):
        group = self.groups.setdefault(sample.dataset_id, {"totals": Counter(), "partitions": {},
            "mbti": {}, "valid_posts_per_user": Counter(), "selected_posts_per_user": Counter(),
            "cleaning_operations": Counter(), "invalid_post_reasons": Counter()})
        values = dict(users=1, label_eligible=int(sample.label_eligible),
            users_with_raw_posts=int(sample.raw_post_count>0), users_with_nonempty_raw_posts=int(sample.nonempty_raw_post_count>0),
            text_eligible=int(sample.text_eligible), supervised_text_eligible=int(sample.sample_eligible),
            users_without_valid_text=int(not sample.text_eligible), raw_posts=sample.raw_post_count,
            users_without_clean_valid_posts=int(sample.valid_post_count==0),
            supervised_text_ineligible=int(sample.label_eligible and not sample.sample_eligible),
            nonempty_raw_posts=sample.nonempty_raw_post_count, valid_posts=sample.valid_post_count,
            candidate_posts=sample.candidate_post_count, selected_posts=len(sample.posts),
            selected_characters=len(sample.text))
        group["totals"].update(values)
        group["partitions"].setdefault(sample.partition or "unassigned", Counter()).update(values)
        group["mbti"].setdefault(sample.mbti or "unlabelled", Counter()).update(values)
        group["valid_posts_per_user"][sample.valid_post_count] += 1
        group["selected_posts_per_user"][len(sample.posts)] += 1
        group["cleaning_operations"].update(dict(sample.operations))
        group["invalid_post_reasons"].update(dict(sample.invalid_post_reasons))


def build_text_manifest(bundle, output, config=TextConfig(), *, limit_per_dataset=None, progress=None):
    """Write selection/qualification metadata and aggregate statistics, never bodies.

    The per-user manifest is a provenance ledger, not a text cache. get_user()
    reconstructs the view from the authenticated source reader.
    """
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    if output.exists():
        raise FileExistsError("Text output already exists; choose a new output directory")
    start, statistics, digest = time.perf_counter(), TextStatistics(), hashlib.sha256()
    free_before = shutil.disk_usage(output.parent).free
    if free_before < 1024**3:
        raise OSError("Insufficient free disk reserve for text provenance")
    with UserTextSource(bundle, config, progress=progress) as source:
        source_digest = file_sha(bundle / "manifest.json")
        output.mkdir(exist_ok=False)
        metadata = {"schema_version": 1, "status": "processing", "readiness_scope": "text_only",
                    "policy": config.policy, "config": asdict(config), "source_bundle": str(bundle),
                    "source_manifest_sha256": source_digest, "source_users_sha256": source.manifest["users_sha256"],
                    "formal_tokenizer_verified": False, "token_eligibility": "not_assessed",
                    "limit_per_dataset": limit_per_dataset, "privacy": "local_private_source_references"}
        write_json(output / "manifest.json", metadata)
        n = 0
        with (output / "text_users.jsonl").open("xb") as stream:
            for sample in source.iter_users(limit_per_dataset=limit_per_dataset):
                statistics.update(sample)
                raw = (json.dumps(sample.provenance(), ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"))+"\n").encode("utf-8")
                stream.write(raw)
                digest.update(raw)
                n += 1
                if progress and n % 100 == 0:
                    progress({"users_processed": n, "pandora_comments_read": source.comments_read})
        for raw_source in (*source.manifest["sources"].values(), source.manifest["comment_source"]):
            check_source(raw_source)
        if file_sha(bundle / "manifest.json") != source_digest:
            raise DataContractError("Raw source manifest changed during text processing")
        expected = len(source.users)
        if limit_per_dataset is None and n != expected:
            raise DataContractError("Incomplete user text coverage")
        metadata.update(status="ready" if limit_per_dataset is None else "partial_audit",
                        users_processed=n, text_users_sha256=digest.hexdigest(),
                        pandora_comments_read=source.comments_read,
                        elapsed_seconds=time.perf_counter()-start, peak_memory_bytes=peak_memory_bytes(),
                        free_disk_before=free_before, free_disk_after=shutil.disk_usage(output).free,
                        python=sys.version, code_sha256=file_sha(Path(__file__)))
        write_json(output / "statistics.json", statistics.groups)
        metadata["statistics_sha256"] = file_sha(output / "statistics.json")
        write_json(output / "manifest.json", metadata)
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit-per-dataset", type=int)
    args = parser.parse_args()
    last = [0.0]

    def progress(info):
        now = time.monotonic()
        if now-last[0] >= 30:
            print(json.dumps({**info, "peak_memory_bytes": peak_memory_bytes()}), flush=True)
            last[0] = now

    print(json.dumps(build_text_manifest(args.bundle, args.output,
        limit_per_dataset=args.limit_per_dataset, progress=progress), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
