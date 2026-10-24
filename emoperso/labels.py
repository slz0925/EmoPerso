"""Canonical four-dimensional MBTI labels and 16-class encoding."""

from numbers import Integral, Real

import numpy as np

DIMENSIONS = ("I/E", "S/N", "T/F", "P/J")
# Each pair is ordered (class 0, class 1), independently of display names.
CLASS_ORDER = (("I", "E"), ("N", "S"), ("T", "F"), ("P", "J"))
MBTI_TYPES = (
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
)


def mbti_to_bits(mbti: str) -> tuple[int, int, int, int]:
    """Encode a four-letter type. Surrounding whitespace/lowercase are accepted."""
    if not isinstance(mbti, str):
        raise ValueError("MBTI must be a four-letter string")
    mbti = mbti.strip().upper()
    if mbti not in MBTI_TYPES:
        raise ValueError(f"Invalid MBTI type: {mbti!r}")
    return tuple(pair.index(letter) for pair, letter in zip(CLASS_ORDER, mbti))


def bits_to_mbti(bits) -> str:
    """Decode exactly four numeric 0/1 values; do not coerce strings/fractions."""
    try:
        values = tuple(bits)
    except TypeError as exc:
        raise ValueError("MBTI bits must contain exactly four numeric 0/1 values") from exc
    if len(values) != 4 or any(
        not isinstance(value, (Real, np.bool_)) or value not in (0, 1) for value in values
    ):
        raise ValueError("MBTI bits must contain exactly four numeric 0/1 values")
    return "".join(pair[int(value)] for pair, value in zip(CLASS_ORDER, values))


def mbti_to_class_id(mbti: str) -> int:
    """Return the original label4 class ID, distinct from a binary encoding."""
    return MBTI_TYPES.index(bits_to_mbti(mbti_to_bits(mbti)))


def class_id_to_mbti(class_id: int) -> str:
    """Decode an integer label4 ID in [0, 15]; booleans are not class IDs."""
    if isinstance(class_id, bool) or not isinstance(class_id, Integral) or not 0 <= class_id < 16:
        raise ValueError("label4 must be an integer class ID from 0 to 15")
    return MBTI_TYPES[int(class_id)]


def bits_to_class_id(bits) -> int:
    return MBTI_TYPES.index(bits_to_mbti(bits))


def class_id_to_bits(class_id: int) -> tuple[int, int, int, int]:
    return mbti_to_bits(class_id_to_mbti(class_id))
