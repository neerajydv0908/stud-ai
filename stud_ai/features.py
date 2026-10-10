"""
Turning a student's record into the numbers the model reads.

The model answers one question for any subject: how likely is it that this
student will NOT pass the next examination in that subject? It only uses what
is known before that exam:

- the subject itself (mathematics, English, science, ...);
- in that subject: attendance %, homework done %, and the test scores so far;
- the other subjects, averaged: their test scores so far, homework % and attendance %.

It never uses the student's name, ID, gender or age, and never the other
subjects' pass/fail results (those are decided at the same time, so they are
not known in advance).

A school year has a different number of tests at different times, so the
features are built from "the first k tests" (k = 1 to 5). During training each
student and subject is shown to the model five times, with 1, 2, 3, 4 and 5
tests known, so it learns to be less sure when it has seen fewer tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# There is no fixed list of subjects: they are read from the CSV's column names (see subjects_in), so a
# file with Hindi, Sanskrit, EVS or social science columns trains the same way as the supplied dataset
# (mathematics, English, science, history, geography). A subject the model wasn't trained on gets no
# estimate ("Not enough data as of now"): inference never submits an unknown category to CatBoost.
MAX_TESTS = 5  # test scores per subject per year that the model reads (the first 1 to 5)

# The columns the model reads, in order. Keep in step with build_features() and features_for_record().
FEATURES: tuple[str, ...] = (
    "subject",
    "attendance",
    "homework",
    "tests_known",
    "latest",
    "average",
    "lowest",
    "highest",
    "trend",
    "other_average",
    "other_homework",
    "other_attendance",
)
CATEGORICAL: tuple[str, ...] = ("subject",)
NUMERIC: tuple[str, ...] = tuple(f for f in FEATURES if f not in CATEGORICAL)
SCORE_FEATURES: tuple[str, ...] = ("tests_known", "latest", "average", "lowest", "highest", "trend")

# Subject names written with a capital letter in sentences ("the next English examination").
_PROPER = {"english", "hindi", "sanskrit", "urdu", "punjabi", "gujarati", "marathi", "bengali", "tamil", "telugu",
           "kannada", "malayalam", "odia", "french", "german", "rajasthani"}


def normalise_subject(subject: str) -> str:
    if not isinstance(subject, str) or len(subject) > 100:
        raise ValueError("The subject must be a short string.")
    s = " ".join(subject.strip().lower().replace("_", " ").split())
    if not s:
        raise ValueError("Give the subject (for example: mathematics, english, science).")
    return {"maths": "mathematics", "math": "mathematics", "evs": "environmental studies"}.get(s, s)


def checked_percent(value: object, what: str, *, optional: bool = True) -> float | None:
    if value is None and optional:
        return None
    # Compare before converting: converting an arbitrarily large int can overflow.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not 0 <= value <= 100:
        raise ValueError(f"{what} must be a percentage between 0 and 100.")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a finite percentage between 0 and 100.")
    return float(value)


def subject_words(subject: str) -> str:
    """How the subject reads inside a sentence: 'mathematics', 'English', 'social science'."""
    s = normalise_subject(subject)
    return " ".join(w.capitalize() if w in _PROPER else w for w in s.split())


def _capital(text: str) -> str:
    """'mathematics tests' -> 'Mathematics tests' (for the start of a sentence)."""
    return text[:1].upper() + text[1:]


def feature_words(feature: str, subject: str) -> str:
    """Plain words for one input, used when explaining a prediction to a teacher."""
    sw = subject_words(subject)
    title = _capital(sw)
    return {
        "subject": "Subject",
        "attendance": f"Attendance in {sw}",
        "homework": f"Homework done in {sw}",
        "scores": f"{title} test scores so far",
        "tests_known": f"{title} tests taken so far",
        "latest": f"Latest {sw} test",
        "average": f"Average of {sw} tests so far",
        "lowest": f"Lowest {sw} test so far",
        "highest": f"Highest {sw} test so far",
        "trend": f"Change in {sw} scores per test",
        "other_average": "Average test score in other subjects",
        "other_homework": "Homework done in other subjects",
        "other_attendance": "Attendance in other subjects",
    }[feature]


def column_prefix(subject: str) -> str:
    """How a subject is written at the start of its CSV columns: 'social science' -> 'social_science'."""
    return normalise_subject(subject).replace(" ", "_")


def test_columns(subject: str) -> list[str]:
    return [f"{column_prefix(subject)}_test_{i}_score" for i in range(1, MAX_TESTS + 1)]


def attendance_column(subject: str) -> str:
    return f"{column_prefix(subject)}_attendance_percent"


def homework_column(subject: str) -> str:
    return f"{column_prefix(subject)}_homework_completion_percent"


def label_column(subject: str) -> str:
    return f"{column_prefix(subject)}_passed"


def subjects_in(columns: Iterable[object]) -> tuple[str, ...]:
    """The subjects a table has, in column order: one for every '<subject>_test_1_score' column."""
    suffix = "_test_1_score"
    found: list[str] = []
    for col in map(str, columns):
        if not col.endswith(suffix) or len(col) == len(suffix):
            continue
        prefix = col[: -len(suffix)]
        subject = normalise_subject(prefix)
        if column_prefix(subject) != prefix:
            raise ValueError(f"Column '{col}': write the subject as '{column_prefix(subject)}' "
                             f"(lower case, '_' between words, full name), e.g. '{column_prefix(subject)}{suffix}'.")
        if subject not in found:
            found.append(subject)
    return tuple(found)


# Every number the model reads is rounded to this many decimals. Without it, rounding noise in the last
# digits (a flat run of scores giving a "trend" of 0.00000000000003 instead of 0) can decide a split in the
# trees, and the same student can get a slightly different risk on another computer or library version.
DECIMALS = 6


def _trends(scores: np.ndarray) -> np.ndarray:
    """Average change per test for each row of a (students x tests) array: the slope of a straight line
    through the scores. Exactly 0 for a flat run of scores, and 0 with one test."""
    k = scores.shape[1]
    if k < 2:
        return np.zeros(len(scores))
    x = np.arange(k, dtype=float) - (k - 1) / 2  # test number, centred: -1, 0, 1 for three tests
    return (scores * x).sum(axis=1) / float((x * x).sum())


@dataclass(frozen=True)
class StudentRecord:
    """What the portal (or a CSV row) knows about one student before the next exam in one subject.

    Percentages and scores are 0-100 (convert marks first: 18 out of 25 = 72). `tests` holds this
    subject's test scores so far, oldest first (1 to 5 of them). `other_tests` holds, for each other
    subject, its test scores so far; or give `other_average` directly. Missing values can be None.
    """

    subject: str
    tests: Sequence[float]
    attendance: float | None = None
    homework: float | None = None
    other_tests: Mapping[str, Sequence[float]] | None = None
    other_average: float | None = None
    other_homework: float | None = None
    other_attendance: float | None = None


@dataclass(frozen=True)
class SubjectMarks:
    """One subject for one student: test scores so far (oldest first, 0-100), attendance % and homework done %."""

    tests: Sequence[float]
    attendance: float | None = None
    homework: float | None = None


def records_for_student(subjects: Mapping[str, SubjectMarks]) -> list[StudentRecord]:
    """One StudentRecord per subject, with the 'other subjects' inputs worked out from the rest of the student's subjects."""
    marks = {normalise_subject(name): m for name, m in subjects.items()}
    if len(marks) != len(subjects):
        raise ValueError("The same subject is given twice (for example 'maths' and 'mathematics').")
    for subject, m in marks.items():
        for name, value in (("attendance", m.attendance), ("homework", m.homework)):
            checked_percent(value, f"{_capital(subject_words(subject))} {name}")
    records = []
    for subject, m in marks.items():
        others = {s: o for s, o in marks.items() if s != subject}
        homework = [o.homework for o in others.values() if o.homework is not None]
        attendance = [o.attendance for o in others.values() if o.attendance is not None]
        records.append(StudentRecord(
            subject=subject, tests=list(m.tests), attendance=m.attendance, homework=m.homework,
            other_tests={s: list(o.tests) for s, o in others.items()},
            other_homework=float(np.mean(homework)) if homework else None,
            other_attendance=float(np.mean(attendance)) if attendance else None,
        ))
    return records


def features_for_record(rec: StudentRecord) -> dict[str, object]:
    """The model's inputs for one student and subject (used by the predictor)."""
    subject = normalise_subject(rec.subject)
    if not 1 <= len(rec.tests) <= MAX_TESTS:
        raise ValueError(f"Give between 1 and {MAX_TESTS} test scores.")
    tests = np.array([checked_percent(s, f"{_capital(subject_words(subject))} test scores", optional=False) for s in rec.tests], dtype=float)
    if not 1 <= len(tests) <= MAX_TESTS:
        raise ValueError(f"Give between 1 and {MAX_TESTS} {subject_words(subject)} test scores (got {len(tests)}).")
    if np.any((tests < 0) | (tests > 100)):
        raise ValueError(f"{_capital(subject_words(subject))} test scores must be between 0 and 100 (convert marks to percentages first).")
    for name, value in (("attendance", rec.attendance), ("homework", rec.homework), ("other_average", rec.other_average),
                        ("other_homework", rec.other_homework), ("other_attendance", rec.other_attendance)):
        checked_percent(value, name)
    other_scores: list[float] = []
    for other, scores in (rec.other_tests or {}).items():
        vals = [checked_percent(s, f"{_capital(subject_words(other))} test scores", optional=False) for s in scores]
        if any(not 0 <= v <= 100 for v in vals):
            raise ValueError(f"{_capital(subject_words(other))} test scores must be between 0 and 100.")
        if vals and normalise_subject(other) != subject:
            other_scores.append(float(np.mean(vals)))
    if not other_scores and rec.other_average is not None:
        other_scores.append(float(rec.other_average))
    nan = float("nan")
    row = {
        "subject": subject,
        "attendance": nan if rec.attendance is None else float(rec.attendance),
        "homework": nan if rec.homework is None else float(rec.homework),
        "tests_known": float(len(tests)),
        "latest": float(tests[-1]),
        "average": float(tests.mean()),
        "lowest": float(tests.min()),
        "highest": float(tests.max()),
        "trend": float(_trends(tests[None, :])[0]),
        "other_average": float(np.mean(other_scores)) if other_scores else nan,
        "other_homework": nan if rec.other_homework is None else float(rec.other_homework),
        "other_attendance": nan if rec.other_attendance is None else float(rec.other_attendance),
    }
    return {f: v if f in CATEGORICAL else float(np.round(v, DECIMALS)) for f, v in row.items()}


def build_features(df: pd.DataFrame, tests_known: Iterable[int] = (MAX_TESTS,), subjects: Sequence[str] | None = None) -> pd.DataFrame:
    """Features for every student in the dataset, for each subject and each number of known tests.

    The subjects are read from the column names (every subject in the file, unless `subjects` picks
    some); the "other subjects" averages always use all the file's other subjects. Returns the feature
    columns plus `student_id`, `k` (tests known, for grouping) and, when the data has it,
    `label_fail` = 1 if the student did NOT pass that subject.
    """
    all_subjects = subjects_in(df.columns)
    if not all_subjects:
        raise ValueError("No subject columns found (expected columns like mathematics_test_1_score).")
    chosen = all_subjects if subjects is None else tuple(normalise_subject(s) for s in subjects)
    missing = [s for s in chosen if s not in all_subjects]
    if missing:
        raise ValueError(f"No test columns for: {', '.join(missing)} (expected e.g. {test_columns(missing[0])[0]}).")
    scores = {s: df[test_columns(s)].to_numpy(dtype=float) for s in all_subjects}
    homework = {s: df[homework_column(s)].to_numpy(dtype=float) for s in all_subjects}
    attendance = {s: df[attendance_column(s)].to_numpy(dtype=float) for s in all_subjects}
    nan = np.full(len(df), np.nan)
    parts = []
    for subject in chosen:
        others = [s for s in all_subjects if s != subject]
        for k in tests_known:
            if not 1 <= k <= MAX_TESTS:
                raise ValueError(f"tests_known must be 1..{MAX_TESTS}")
            own = scores[subject][:, :k]
            trend = _trends(own)
            part = pd.DataFrame(
                {
                    "subject": subject,
                    "attendance": attendance[subject],
                    "homework": homework[subject],
                    "tests_known": float(k),
                    "latest": own[:, -1],
                    "average": own.mean(axis=1),
                    "lowest": own.min(axis=1),
                    "highest": own.max(axis=1),
                    "trend": trend,
                    "other_average": np.mean([scores[o][:, :k].mean(axis=1) for o in others], axis=0) if others else nan,
                    "other_homework": np.mean([homework[o] for o in others], axis=0) if others else nan,
                    "other_attendance": np.mean([attendance[o] for o in others], axis=0) if others else nan,
                }
            )
            part[list(NUMERIC)] = part[list(NUMERIC)].round(DECIMALS)
            part.insert(0, "k", k)
            part.insert(0, "student_id", df["student_id"].to_numpy())
            if label_column(subject) in df.columns:
                part["label_fail"] = 1 - df[label_column(subject)].to_numpy(dtype=int)
            parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    assert tuple(c for c in out.columns if c in FEATURES) == FEATURES
    return out


def record_from_row(row: pd.Series, subject: str, tests_known: int = MAX_TESTS) -> StudentRecord:
    """A StudentRecord from one row of the dataset (first `tests_known` tests of each subject)."""
    subject = normalise_subject(subject)
    others = [s for s in subjects_in(row.index) if s != subject]
    return StudentRecord(
        subject=subject,
        tests=[float(row[c]) for c in test_columns(subject)[:tests_known]],
        attendance=float(row[attendance_column(subject)]),
        homework=float(row[homework_column(subject)]),
        other_tests={s: [float(row[c]) for c in test_columns(s)[:tests_known]] for s in others},
        other_homework=float(np.mean([float(row[homework_column(s)]) for s in others])) if others else None,
        other_attendance=float(np.mean([float(row[attendance_column(s)]) for s in others])) if others else None,
    )
