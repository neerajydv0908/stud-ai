"""
Predict a student's risk of not passing the next examination in a subject.

From Python, every subject of one student at once (the "other subjects" inputs are worked out for you):

    from stud_ai.features import SubjectMarks
    from stud_ai.predict import predict_student

    for result in predict_student({
        "mathematics": SubjectMarks(tests=[45, 38, 41], attendance=72, homework=55),
        "english": SubjectMarks(tests=[62, 58, 66], attendance=90, homework=80),
        "science": SubjectMarks(tests=[50, 47, 52], attendance=85, homework=70),
    }):
        print(result.text)        # Estimated risk of not passing the next mathematics examination: NN%
        for r in result.reasons:  # the things that moved the estimate most
            print("  -", r.sentence)

Or one subject: predict(StudentRecord(subject="science", tests=[...], attendance=..., homework=..., other_average=...)).

From the command line:

    python -m stud_ai.predict --subject science --tests 45 38 41 --attendance 72 --homework 55 --other-average 52
    python -m stud_ai.predict --csv test_students.csv --tests-known 5 --out predictions.csv     # every subject
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from catboost import Pool

from .features import (
    CATEGORICAL, FEATURES, MAX_TESTS, NUMERIC, SCORE_FEATURES, StudentRecord, SubjectMarks, build_features, feature_words,
    features_for_record, normalise_subject, records_for_student, subject_words,
)
from .model import LoadedModel, load_model

SENTENCE = "Estimated risk of not passing the next {subject} examination: {percent}%"
# A subject the model was never trained on gets no estimate, only this.
NOT_ENOUGH_DATA = "Not enough data as of now for {subject}"

_cached: LoadedModel | None = None


def _model() -> LoadedModel:
    global _cached
    if _cached is None:
        _cached = load_model()
    return _cached


def risk_percent(risk: float) -> int:
    """Whole percent, kept between 1% and 99%: the model is never certain."""
    return int(min(99, max(1, round(risk * 100))))


def sentence(subject: str, percent: int) -> str:
    return SENTENCE.format(subject=subject_words(subject), percent=percent)


def not_enough_data(subject: str) -> str:
    return NOT_ENOUGH_DATA.format(subject=subject_words(subject))


@dataclass
class Reason:
    key: str  # "scores" or one feature name
    words: str
    value_text: str
    effect: str  # "raises" or "lowers"
    weight: float  # summed SHAP value in log-odds (bigger = moved the estimate more)

    @property
    def sentence(self) -> str:
        return f"{self.words}: {self.value_text} ({self.effect} the risk)"


@dataclass
class RiskResult:
    subject: str
    risk: float | None  # 0-1, chance of NOT passing; None for a subject the model wasn't trained on
    percent: int | None  # the risk as a whole percent (1-99); None for a subject the model wasn't trained on
    text: str
    reasons: list[Reason] = field(default_factory=list)
    assumed: list[str] = field(default_factory=list)  # inputs that were missing and filled with a typical value
    known_subject: bool = True  # False: not trained on this subject, so no estimate ("Not enough data as of now")


# Contributions smaller than this (in log-odds) are too small to be worth showing a teacher.
MIN_WEIGHT = 0.05


def _value_text(feature: str, value: float) -> str:
    if feature.endswith(("attendance", "homework")):
        return f"{value:.0f}%"
    return f"{value:.0f} out of 100"


def _scores_text(row: pd.Series) -> str:
    n = int(row["tests_known"])
    if n == 1:
        return f"{row['latest']:.0f} out of 100 in the only test so far"
    text = f"average {row['average']:.0f}, latest {row['latest']:.0f} (out of 100, {n} tests)"
    if abs(row["trend"]) >= 2:
        text += f", {'rising' if row['trend'] > 0 else 'falling'} about {abs(row['trend']):.0f} marks per test"
    return text


def _explain(loaded: LoadedModel, X: pd.DataFrame, top: int, skips: Sequence[set[str]] | None = None) -> list[list[Reason]]:
    """The inputs that moved each student's estimate most (CatBoost's own SHAP values); this subject's scores count as one.
    `skips[i]` holds the inputs that were filled in for row i: they are never given as reasons."""
    shap = loaded.model.get_feature_importance(data=Pool(X, cat_features=list(CATEGORICAL)), type="ShapValues")[:, :-1]
    idx = {f: j for j, f in enumerate(FEATURES)}
    out = []
    for i in range(len(X)):
        row = X.iloc[i]
        subject = row["subject"]
        skip = skips[i] if skips is not None else frozenset()
        reasons = [Reason("scores", feature_words("scores", subject), _scores_text(row), "",
                          float(sum(shap[i, idx[f]] for f in SCORE_FEATURES)))]
        for f in NUMERIC:
            if f not in SCORE_FEATURES and f not in skip:  # never "explain" with a value we filled in ourselves
                reasons.append(Reason(f, feature_words(f, subject), _value_text(f, float(row[f])), "", float(shap[i, idx[f]])))
        reasons = [r for r in sorted(reasons, key=lambda r: -abs(r.weight)) if abs(r.weight) >= MIN_WEIGHT][:top]
        for r in reasons:
            r.effect = "raises" if r.weight > 0 else "lowers"
        out.append(reasons)
    return out


def _fill_missing(row: dict, card: dict) -> tuple[dict, list[str]]:
    """Fill numbers we don't have with the typical (median) training value; return the inputs filled in."""
    medians = card.get("imputation_medians", {})
    filled = []
    for f in NUMERIC:
        if isinstance(row[f], float) and math.isnan(row[f]):
            if f not in medians:
                raise ValueError(f"{f} is missing and the model card has no typical value for it.")
            row[f] = medians[f]
            filled.append(f)
    return row, filled


def predict_records(records: Sequence[StudentRecord], top_reasons: int = 3, loaded: LoadedModel | None = None) -> list[RiskResult]:
    """Risks for many students and subjects at once, with the main reasons (one model call, so it is fast).

    A subject the model wasn't trained on never reaches CatBoost: it gets no estimate, only
    "Not enough data as of now".
    """
    raw_rows = [features_for_record(record) for record in records]
    if not raw_rows:
        return []
    loaded = loaded or _model()
    trained = set(loaded.card["subjects_trained_on"])
    rows, fills, indices = [], [], []
    results: list[RiskResult | None] = [None] * len(records)
    for i, raw in enumerate(raw_rows):
        subject = raw["subject"]
        if subject not in trained:
            results[i] = RiskResult(subject=subject, risk=None, percent=None, text=not_enough_data(subject), known_subject=False)
            continue
        row, filled = _fill_missing(raw, loaded.card)
        rows.append(row)
        fills.append(set(filled))
        indices.append(i)
    if rows:
        X = pd.DataFrame(rows, columns=list(FEATURES))
        risks = loaded.model.predict_proba(X)[:, 1]
        explanations = _explain(loaded, X, top_reasons, fills) if top_reasons else [[] for _ in rows]
        for i, row, filled, risk, reasons in zip(indices, rows, fills, risks, explanations):
            risk = float(risk)
            if not math.isfinite(risk) or not 0 <= risk <= 1:
                raise ValueError("The model returned an invalid probability.")
            subject, pct = row["subject"], risk_percent(risk)
            results[i] = RiskResult(subject=subject, risk=risk, percent=pct, text=sentence(subject, pct), reasons=reasons,
                assumed=[feature_words(f, subject) for f in NUMERIC if f in filled], known_subject=True)
    return results


def predict(record: StudentRecord, top_reasons: int = 3, loaded: LoadedModel | None = None) -> RiskResult:
    """The risk for one student in one subject, with the main reasons."""
    return predict_records([record], top_reasons, loaded)[0]


def predict_student(subjects: Mapping[str, SubjectMarks], top_reasons: int = 3, loaded: LoadedModel | None = None) -> list[RiskResult]:
    """The risk in every subject for one student, in the order given."""
    return predict_records(records_for_student(subjects), top_reasons, loaded)


def predict_frame(df: pd.DataFrame, tests_known: int = MAX_TESTS, subjects=None, loaded: LoadedModel | None = None) -> pd.DataFrame:
    """Risks for every student and subject in a dataset-shaped table (or only `subjects`), using their first `tests_known` tests."""
    loaded = loaded or _model()
    F = build_features(df, (tests_known,), subjects)
    unknown = set(F["subject"]) - set(loaded.card["subjects_trained_on"])
    if unknown:
        raise ValueError(f"Not enough data as of now for {', '.join(sorted(map(subject_words, unknown)))}: the model hasn't been trained on "
                         "these subjects yet. Retrain it with them first.")
    risk = loaded.model.predict_proba(F[list(FEATURES)])[:, 1]
    out = pd.DataFrame({"student_id": F["student_id"], "subject": F["subject"], "tests_known": tests_known,
                        "risk": risk.round(4), "risk_percent": [risk_percent(r) for r in risk]})
    out["text"] = [sentence(s, p) for s, p in zip(out["subject"], out["risk_percent"])]
    if "label_fail" in F:
        out["actually_not_passed"] = F["label_fail"].to_numpy()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Estimate the risk of not passing the next examination in a subject.")
    ap.add_argument("--subject", help="the subject, for example mathematics, english, science")
    ap.add_argument("--tests", type=float, nargs="+", help="that subject's test scores so far, oldest first, as percentages (1-5 of them)")
    ap.add_argument("--attendance", type=float, help="attendance %% (in that subject, or overall)")
    ap.add_argument("--homework", type=float, help="homework done in that subject, %%")
    ap.add_argument("--other-average", type=float, help="average test score in the other subjects, %%")
    ap.add_argument("--other-homework", type=float, help="homework done in the other subjects, %%")
    ap.add_argument("--other-attendance", type=float, help="attendance %% in the other subjects")
    ap.add_argument("--csv", type=Path, help="score every student (and subject) in a dataset-shaped CSV instead")
    ap.add_argument("--tests-known", type=int, default=MAX_TESTS, help="with --csv: how many tests to use (1-5)")
    ap.add_argument("--out", type=Path, help="with --csv: write the predictions here")
    args = ap.parse_args(argv)

    if args.csv:
        subjects = [normalise_subject(args.subject)] if args.subject else None  # None = every subject in the file
        result = predict_frame(pd.read_csv(args.csv), args.tests_known, subjects)
        if args.out:
            result.to_csv(args.out, index=False)
            print(f"Wrote {len(result)} predictions to {args.out}")
        else:
            print(result.head(20).to_string(index=False))
        return 0
    if not args.subject or not args.tests:
        ap.error("give --subject and --tests (or --csv)")
    res = predict(StudentRecord(subject=args.subject, tests=args.tests, attendance=args.attendance, homework=args.homework,
                                other_average=args.other_average, other_homework=args.other_homework, other_attendance=args.other_attendance))
    print(res.text)
    for r in res.reasons:
        print(f"  - {r.sentence}")
    if res.assumed:
        print("  (not given, so a typical value was used: " + ", ".join(res.assumed) + ")")
    if not res.known_subject:
        print(f"  (no prediction: the model hasn't been trained on {subject_words(res.subject)} yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
