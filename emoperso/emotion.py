"""Deterministic English emotion cues and multi-label auxiliary targets.

Scores describe lexical evidence, not calibrated probabilities. Valid rows use
closed-world targets: detected categories=1, other configured categories=0.
Neutral/negated rows have valid=False and do not supervise the emotion head.
"""
import math
import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np

from .errors import DataContractError
from .token_inputs import batch_signature, _integer

DEFAULT_EMOTIONS = ("joy", "anger", "sadness", "fear", "disgust", "surprise", "contempt")
RULE_VERSION = "english-affective-cues-v2"

# Explicit affective terms; broad topical terms such as 'death' and 'danger'
# alone are deliberately insufficient to assert a user's emotion.
_CUES = {
    "joy": "happy|happiness|joy|joyful|delighted|delightful|elated|cheerful|excited|"
           "thrilled|ecstatic|glad|grateful|thankful|overjoyed|blissful|love this|"
           "love it|love|loved|loving|adore|adored|on cloud nine|made my day|full of joy|looking forward to",
    "anger": "angry|anger|furious|enraged|irate|livid|annoyed|annoying|irritated|"
             "irritating|frustrated|frustrating|outraged|resentful|infuriated|"
             "infuriating|pissed off|fed up|makes my blood boil|lost my temper|so mad",
    "sadness": "sad|sadness|saddened|unhappy|miserable|heartbroken|heartbreaking|"
               "devastated|dejected|despondent|hopeless|lonely|gloomy|sorrow|sorrowful|"
               "grieving|grief|tearful|crying|depressed|depression|in tears|feel down|feeling down|broken heart",
    "fear": "afraid|fearful|scared|scary|terrified|terrifying|frightened|frightening|"
            "anxious|anxiety|worried|worrying|nervous|panicked|panicking|petrified|"
            "uneasy|apprehensive|dreading|worry|worries|concerned|freaking out|on edge|fear for|filled with dread",
    "disgust": "disgusted|disgusting|disgust|revolting|repulsed|repulsive|sickening|"
               "nauseating|nauseated|vile|loathsome|abhorrent|repugnant|yuck|eww|"
               "grossed out|makes me sick|makes my skin crawl|turns my stomach|"
               "can't stomach|cannot stomach|hate|hated|hating|loathe|loathed|loathing",
    "surprise": "surprised|surprising|astonished|astonishing|amazed|stunned|"
                "startled|startling|shocked|shocking|astounded|astounding|"
                "flabbergasted|gobsmacked|speechless|taken aback|out of the blue|"
                "can't believe|cannot believe|didn't expect|did not expect",
    "contempt": "contempt|contemptuous|disdain|disdainful|scorn|scornful|sneering|"
                "despicable|pathetic|ridiculous|laughable|beneath me|beneath contempt|"
                "look down on|looking down on|looked down on|what a loser|utter fool|not worth my time|"
                "have no respect for|no respect for|couldn't care less",
}
_WORD = re.compile(r"\w+(?:'\w+)?", re.UNICODE)
_CLAUSE = re.compile(r"[.!?;,:\n]+|\b(?:but|however|yet|although|because|whereas)\b|"
                     r"\b(?:and|or)\s+(?=(?:i|we|you|he|she|they|it)\b)", re.IGNORECASE)
_NEGATIONS = frozenset(("not", "no", "never", "neither", "nor", "without", "hardly", "barely",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "can't", "cannot",
    "couldn't", "won't", "wouldn't", "shouldn't", "hasn't", "haven't", "hadn't"))
_INTENSIFIERS = frozenset(("very", "really", "extremely", "incredibly", "so", "deeply", "utterly",
                         "absolutely", "totally", "truly", "terribly", "intensely"))
# A small grammar of transparent complements, not an enlarged distance window.
# It handles e.g. "not feel even remotely ..." and "can't say I am ..." while
# lexical content/conjunctions stop its scope. The ordinary short window remains
# configurable for direct negation; setting it to zero disables both mechanisms.
_COMPLEMENT = frozenset(("i", "we", "you", "he", "she", "they", "it", "am", "is", "are", "was", "were",
    "be", "been", "being", "do", "does", "did", "to", "feel", "feels", "feeling", "felt", "say", "says",
    "said", "claim", "think", "believe", "seem", "seems", "get", "even", "remotely", "at", "all",
    "particularly", "quite", "slightly", "somewhat", "reason", "any", "a", "bit")) | _INTENSIFIERS
_AFFECTIVE_NEGATIVE_PHRASES = frozenset((
    ("can't", "believe"), ("cannot", "believe"), ("didn't", "expect"), ("did", "not", "expect"),
    ("can't", "stomach"), ("cannot", "stomach"), ("couldn't", "care", "less"),
    ("no", "respect", "for"), ("have", "no", "respect", "for"), ("not", "worth", "my", "time")))
_EMOTION_NOUNS = frozenset(("happiness", "joy", "anger", "sadness", "grief", "sorrow", "anxiety",
                           "depression", "disgust", "contempt", "scorn"))
_EXPERIENCE = frozenset(("feel", "feels", "feeling", "felt", "have", "having", "had", "experience",
    "experiencing", "experienced", "suffer", "suffering", "overcome", "overwhelmed", "filled"))
_PEOPLE = frozenset(("people", "others", "them", "him", "her", "you", "anyone", "everyone", "colleagues",
                    "friends", "family", "cowards", "liars", "idiots"))
_PHYSICAL = frozenset(("street", "road", "balcony", "window", "roof", "bridge", "valley", "ground", "below", "stairs"))
_AFFECT_VERBS = frozenset(("love", "loved", "loving", "adore", "adored", "hate", "hated", "hating", "loathe", "loathed", "loathing"))
_TOPIC = frozenset(("read", "reading", "article", "articles", "about", "discuss", "discussed", "discussing",
                    "definition", "word", "words", "term", "terms", "topic"))


def _negated(words, position, protected, window):
    if not window:
        return False
    transparent = True
    for j in range(position-1, -1, -1):
        word = words[j]
        exempt = word == "not" and j+1 < len(words) and words[j+1] in ("only", "just")
        if word in _NEGATIONS and j not in protected and not exempt:
            if position-j <= window or transparent:
                return True
        transparent = transparent and word in _COMPLEMENT
        if position-j >= window and not transparent:
            break
    return False


def _context_ok(words, start, end, cue):
    if len(cue) == 1 and cue[0] in _AFFECT_VERBS:
        before = words[max(0,start-5):start]
        topic = [i for i,w in enumerate(before) if w in _TOPIC]
        if topic and not set(before[topic[-1]+1:]) & {"i", "we", "me", "us"}:
            return False
    if cue in (("look", "down", "on"), ("looking", "down", "on"), ("looked", "down", "on")):
        nearby = words[max(0,start-4):end+10]
        obj = words[end:end+4]
        return bool(_PEOPLE.intersection(obj)) and not _PHYSICAL.intersection(nearby)
    if len(cue) == 1 and cue[0] in _EMOTION_NOUNS:
        before = words[max(0,start-5):start]
        after = words[end:end+5]
        # Topic nouns need an experiencer, not just "read about anxiety".
        experience = any(w in _EXPERIENCE and all(v in _COMPLEMENT or v in
                         ("such", "this", "that", "overwhelming", "intense", "constant", "with")
                         for v in before[j+1:]) for j,w in enumerate(before))
        return (bool(before and before[-1] in ("my", "our"))
                or experience
                or bool(set(after) & {"me", "my", "us", "our"}))
    return True


def emotion_categories(values):
    if isinstance(values, str):
        raise ValueError("categories must be an ordered sequence, not a string")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("categories must be an ordered nonempty sequence") from exc
    if (not result or any(not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", v) for v in result)
            or len(set(result)) != len(result)):
        raise ValueError("Emotion categories must be distinct lowercase names")
    return result


def _normalize(text):
    return unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'").casefold()


@dataclass(frozen=True)
class EmotionEvidence:
    targets: np.ndarray
    valid: bool
    scores: np.ndarray
    categories: tuple[str, ...]


class EmotionLabeler:
    """Configurable phrase rules with clause/post-local negation.

    Each cue scores 1, a preceding intensifier multiplies by 1.5, and each
    exclamation mark in that clause adds .1 (at most .3). Category evidence is
    the maximum cue score, not a count of repeated words. A cue is suppressed
    using position-aware affective phrases and transparent negative complements,
    with `negation_window` as a fallback for direct local negation. 'Not only'
    and 'not just' are exempt. Negation never invents an opposite emotion.
    Custom categories require explicit nonempty cue sequences in `cues`.
    """
    def __init__(self, categories=DEFAULT_EMOTIONS, *, cues=None, threshold=1.0, negation_window=3):
        self.categories = emotion_categories(categories)
        if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if not _integer(negation_window) or negation_window > 16:
            raise ValueError("negation_window must be in [0,16]")
        source = {key: value.split("|") for key, value in _CUES.items()}
        if cues is not None:
            if not isinstance(cues, dict) or any(key not in self.categories for key in cues):
                raise ValueError("cues must map configured category names to phrase sequences")
            source.update(cues)
        normalized = {}
        for category in self.categories:
            phrases = source.get(category)
            if not isinstance(phrases, (tuple, list)) or not phrases:
                raise ValueError(f"Provide nonempty cues for emotion {category}")
            tokens = []
            for phrase in phrases:
                if not isinstance(phrase, str) or not phrase.strip():
                    raise ValueError("Each emotion cue must be a nonempty phrase")
                words = tuple(_WORD.findall(_normalize(phrase)))
                if not words:
                    raise ValueError("Emotion cues must contain words")
                tokens.append(words)
            normalized[category] = tuple(sorted(set(tokens)))
        self.cues = MappingProxyType(normalized)
        index = {}
        for c, category in enumerate(self.categories):
            for cue in self.cues[category]:
                index.setdefault(cue[0], []).append((c, cue))
        self._index = MappingProxyType({key:tuple(values) for key,values in index.items()})
        # Explicit custom cues are literal rules chosen by the caller. Context
        # filters below apply only to the project defaults, not custom lexicons.
        self._context_categories = frozenset(c for c in self.categories if c not in (cues or {}))
        self.threshold, self.negation_window = float(threshold), int(negation_window)

    def _matches(self, words):
        """Position/category/phrase matches; only first-word candidates are tested."""
        for i, word in enumerate(words):
            for c, cue in self._index.get(word, ()):
                if tuple(words[i:i+len(cue)]) == cue:
                    yield i, c, cue

    def label_posts(self, posts):
        """Score separate text units; phrase and negation matching never cross one."""
        if isinstance(posts, str):
            raise TypeError("Pass a sequence of posts, or use label_text")
        scores = np.zeros(len(self.categories), dtype=np.float32)
        for post in posts:
            if not isinstance(post, str):
                raise TypeError("Posts must be strings")
            text = _normalize(post)
            start = 0
            # Keep trailing punctuation attached for evidence strength.
            for boundary in list(_CLAUSE.finditer(text)) + [None]:
                end = boundary.start() if boundary else len(text)
                clause = text[start:end]
                punctuation = boundary.group() if boundary else ""
                words = _WORD.findall(clause)
                matches = tuple(self._matches(words))
                protected = {j for i, _, cue in matches if cue in _AFFECTIVE_NEGATIVE_PHRASES
                             for j in range(i,i+len(cue)) if words[j] in _NEGATIONS}
                for i, c, cue in matches:
                    if _negated(words, i, protected, self.negation_window):
                        continue
                    if self.categories[c] in self._context_categories and not _context_ok(words,i,i+len(cue),cue):
                        continue
                    strength = 1.5 if any(w in _INTENSIFIERS for w in words[max(0,i-2):i]) else 1.
                    strength += min(punctuation.count("!"), 3)*.1
                    scores[c] = max(scores[c], strength)
                if boundary:
                    start = boundary.end()
        targets = (scores >= self.threshold).astype(np.float32)
        return EmotionEvidence(targets, bool(targets.any()), scores, self.categories)

    def label_text(self, text):
        return self.label_posts([text])

    def label_batch(self, batch):
        """Return targets [B,C], valid [B], from certified retained content only."""
        batch_signature(batch)
        evidence = [self.label_posts(retained_posts(row)) for row in batch["provenance"]]
        return np.stack([e.targets for e in evidence]), np.asarray([e.valid for e in evidence], dtype=bool)


def retained_posts(row):
    """Intersect retained non-special character spans with real source posts.

    Partial words at span boundaries are omitted from emotion matching.
    Gaps/specials and true posts are separate rule scopes. Generic
    adapters without certified offsets require explicit external targets.
    """
    end, spans, text = row.retained_characters, row.content_character_spans, row.source.text
    if not _integer(end) or end > len(text) or not isinstance(spans, tuple):
        raise DataContractError("Automatic emotion targets require certified retained character spans; use the formal tokenizer or explicit external targets")
    previous = 0
    for pair in spans:
        if (not isinstance(pair, tuple) or len(pair) != 2 or not all(_integer(v) for v in pair)
                or not previous <= pair[0] < pair[1] <= end):
            raise DataContractError("Invalid retained content character spans")
        previous = pair[1]
    posts = [(p.user_text_start, p.user_text_end) for p in row.source.posts] or [(0,len(text))]
    chunks = []
    for pstart, pend in posts:
        for a, b in spans:
            left, right = max(pstart, a), min(pend, b)
            if left >= right:
                continue
            # A cutoff inside a word must not create a standalone lexical cue.
            wordchar = lambda c: c.isalnum() or c == "_"
            while left < right and left > pstart and wordchar(text[left-1]) and wordchar(text[left]):
                left += 1
            while right > left and right < pend and wordchar(text[right-1]) and wordchar(text[right]):
                right -= 1
            if left < right:
                chunks.append(text[left:right])
    return chunks
