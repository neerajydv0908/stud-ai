"""
Train the Stud-AI risk model (one model for every subject).

    python -m stud_ai.train            # train on train_students.csv, stop early on validation_students.csv
    python -m stud_ai.train --search   # first try a small grid of settings on the validation set

The test file is NOT used here; run `python -m stud_ai.evaluate` once you are happy.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .features import (
    CATEGORICAL, FEATURES, MAX_TESTS, NUMERIC, attendance_column, build_features, homework_column, label_column, subjects_in,
    test_columns,
)
from . import __version__
from .metrics import expected_calibration_error, summary
from .model import CARD_FILE, MODEL_DIR, MODEL_FILE, MONOTONE, PARAMS, REPORT_DIR, THRESHOLD, data_file, new_model

TESTS_KNOWN = tuple(range(1, MAX_TESTS + 1))


def required_columns(subjects) -> list[str]:
    cols = ["student_id"]
    for s in subjects:
        cols += [attendance_column(s), homework_column(s), *test_columns(s), label_column(s)]
    return cols


def load_dataset(path: Path) -> pd.DataFrame:
    """Read one CSV and check it has what the model needs. The subjects are read from its column names."""
    df = pd.read_csv(path)
    subjects = subjects_in(df.columns)
    if len(subjects) < 2:
        raise ValueError(f"{path.name}: found {len(subjects)} subject(s); the model needs at least two "
                         "(columns like mathematics_test_1_score ... mathematics_test_5_score, mathematics_passed).")
    required = required_columns(subjects)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if df["student_id"].duplicated().any():
        raise ValueError(f"{path.name}: some student_id values appear more than once.")
    blanks = df[required].isna().sum()
    if blanks.any():
        raise ValueError(f"{path.name} has empty cells in: " + ", ".join(f"{c} ({n})" for c, n in blanks[blanks > 0].items()))
    for s in subjects:
        if not set(df[label_column(s)].unique()) <= {0, 1}:
            raise ValueError(f"{path.name}: {label_column(s)} must be 0 or 1")
    numeric = [c for c in required if c != "student_id" and not c.endswith("_passed")]
    out_of_range = [c for c in numeric if (df[c] < 0).any() or (df[c] > 100).any()]
    if out_of_range:
        raise ValueError(f"{path.name}: values outside 0-100 in {', '.join(out_of_range)}")
    return df


def same_subjects(*frames: tuple[str, pd.DataFrame]) -> tuple[str, ...]:
    """The files must have the same subjects; returns them."""
    first_name, first = frames[0]
    subjects = subjects_in(first.columns)
    for name, df in frames[1:]:
        if set(subjects_in(df.columns)) != set(subjects):
            raise ValueError(f"{name} has subjects {list(subjects_in(df.columns))}, but {first_name} has {list(subjects)}.")
    return subjects


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def breakdown(frame: pd.DataFrame, risk: np.ndarray, subjects) -> dict:
    """Results by number of tests known (all subjects) and by subject (5 tests known)."""
    y = frame["label_fail"].to_numpy()
    by_k = []
    for k in TESTS_KNOWN:
        m = frame["k"].to_numpy() == k
        s = summary(y[m], risk[m], THRESHOLD)
        by_k.append({"tests_known": k, **{key: s[key] for key in ("roc_auc", "log_loss", "brier", "accuracy", "recall", "precision")}})
    by_subject = []
    k5 = frame["k"].to_numpy() == MAX_TESTS
    for subj in subjects:
        m = k5 & (frame["subject"].to_numpy() == subj)
        s = summary(y[m], risk[m], THRESHOLD)
        by_subject.append({"subject": subj, **{key: s[key] for key in ("students", "actual_not_passed", "roc_auc", "brier", "accuracy", "recall", "precision")}})
    return {"by_tests_known": by_k, "by_subject_5_tests": by_subject}


def search(train: pd.DataFrame, val: pd.DataFrame) -> list[dict]:
    """A small grid over depth, learning rate and L2, with and without the 'higher marks never raise the risk' rules."""
    X, y = train[list(FEATURES)], train["label_fail"]
    Xv, yv = val[list(FEATURES)], val["label_fail"]
    rows = []
    for depth, lr, l2, mono in itertools.product([4, 6, 8], [0.05, 0.1], [3, 30], [False, True]):
        over = {"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2}
        m = new_model(**over) if mono else CatBoostClassifier(**{**PARAMS, **over}, cat_features=list(CATEGORICAL))
        m.fit(X, y, eval_set=(Xv, yv), use_best_model=True)
        s = summary(yv.to_numpy(), m.predict_proba(Xv)[:, 1])
        rows.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2, "monotone_rules": mono,
                     "trees": m.get_best_iteration() + 1, "log_loss": s["log_loss"], "roc_auc": s["roc_auc"], "brier": s["brier"]})
        print(f"  depth {depth} lr {lr} l2 {l2} rules {'on ' if mono else 'off'} → log loss {s['log_loss']:.4f}, AUC {s['roc_auc']:.4f}", flush=True)
    return sorted(rows, key=lambda r: r["log_loss"])


def write_search_report(rows: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Settings tried on the validation set",
        "",
        "Each setting was trained on `train_students.csv` (every student, every subject, with 1–5 tests known) and scored on",
        "`validation_students.csv` the same way. Lower log loss = better probabilities. The test file was not used.",
        "Every setting reads the subject as plain yes/no columns (`one_hot_max_size=10`), which keeps the 'never raise' rules intact.",
        "",
        "| Depth | Learning rate | L2 | 'Higher marks never raise the risk' | Trees | Log loss | ROC-AUC | Brier |",
        "| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(f"| {r['depth']} | {r['learning_rate']} | {r['l2_leaf_reg']} | {'yes' if r['monotone_rules'] else 'no'} | {r['trees']} | "
                     f"{r['log_loss']:.4f} | {r['roc_auc']:.4f} | {r['brier']:.4f} |")
    used = next((r for r in rows if r["monotone_rules"] and r["depth"] == PARAMS["depth"] and r["learning_rate"] == PARAMS["learning_rate"]
                 and r["l2_leaf_reg"] == PARAMS["l2_leaf_reg"]), None)
    if used:
        best, gap = rows[0], used["log_loss"] - rows[0]["log_loss"]
        lines += ["", f"**Used:** depth {used['depth']}, learning rate {used['learning_rate']}, L2 {used['l2_leaf_reg']}, rules on "
                      f"(log loss {used['log_loss']:.4f}; the best above is {best['log_loss']:.4f})."]
        if gap < 0.001:
            lines += ["A gap this small (under 0.001) is within chance: scoring the same model on a different set of students moves its "
                      "log loss by much more than that. So the settings in `stud_ai/model.py` are kept."]
        else:
            lines += [f"The best setting is {gap:.4f} better. Consider using it in `stud_ai/model.py`, then check it with `experiments/cv.py`."]
    path = REPORT_DIR / "validation-search.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Stud-AI risk model (CatBoost, every subject).")
    ap.add_argument("--data", type=Path, help="folder with the CSV files (default: data/ when it holds them, else the project folder)")
    ap.add_argument("--search", action="store_true", help="try a grid of settings on the validation set first")
    args = ap.parse_args(argv)

    train_path, val_path = data_file("train", args.data), data_file("validation", args.data)
    train_df, val_df = load_dataset(train_path), load_dataset(val_path)
    subjects = same_subjects((train_path.name, train_df), (val_path.name, val_df))
    overlap = set(train_df["student_id"]) & set(val_df["student_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} students are in both the training and validation files.")

    train = build_features(train_df, TESTS_KNOWN, subjects)
    val = build_features(val_df, TESTS_KNOWN, subjects)
    print(f"Subjects: {', '.join(subjects)}")
    print(f"Training students: {len(train_df)} x {len(subjects)} subjects ({len(train)} rows with 1–{MAX_TESTS} tests known); "
          f"validation students: {len(val_df)}; not passed: {train['label_fail'].mean():.1%} / {val['label_fail'].mean():.1%} of subject results")

    if args.search:
        print("Trying settings on the validation set…")
        rows = search(train, val)
        print(f"Wrote {write_search_report(rows)}")

    t0 = time.time()
    model = new_model()
    model.fit(train[list(FEATURES)], train["label_fail"], eval_set=(val[list(FEATURES)], val["label_fail"]), use_best_model=True)
    risk = model.predict_proba(val[list(FEATURES)])[:, 1]
    overall = summary(val["label_fail"].to_numpy(), risk, THRESHOLD)
    parts = breakdown(val, risk, subjects)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))
    card = {
        "name": "Stud-AI risk model",
        "question": "How likely is it that the student will NOT pass the next examination in a given subject?",
        "output": "risk = probability of not passing, 0-1; shown as 'Estimated risk of not passing the next <subject> examination: NN%'",
        "model": "CatBoostClassifier (gradient-boosted decision trees), one model for every subject",
        "subjects_trained_on": list(subjects),
        "features": list(FEATURES),
        "categorical_features": list(CATEGORICAL),
        "monotone_constraints": {f: MONOTONE[f] for f in FEATURES},
        "not_used": ["student_id", "sr_no", "age", "gender", "other subjects' pass/fail (decided at the same time, so unknown beforehand)"],
        "params": {k: v for k, v in PARAMS.items() if k not in ("verbose", "allow_writing_files")},
        "trees": model.get_best_iteration() + 1,
        "threshold": THRESHOLD,
        "imputation_medians": {f: float(np.nanmedian(train[f])) for f in NUMERIC},
        "training_data": {"file": train_path.name, "sha256": sha256(train_path), "students": len(train_df)},
        "validation_data": {"file": val_path.name, "sha256": sha256(val_path), "students": len(val_df)},
        "validation": {"all_subjects_all_tests_known": overall, **parts,
                       "calibration_error": expected_calibration_error(val["label_fail"].to_numpy(), risk)},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"stud_ai": __version__, "python": platform.python_version(), "catboost": catboost.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__},
    }
    CARD_FILE.write_text(json.dumps(card, indent=2), encoding="utf-8")

    k5 = parts["by_tests_known"][-1]
    print(f"Trained {card['trees']} trees in {time.time() - t0:.1f}s → {MODEL_FILE.relative_to(MODEL_FILE.parent.parent)}")
    print(f"Validation, 5 tests known, all subjects: ROC-AUC {k5['roc_auc']:.3f}, accuracy {k5['accuracy']:.1%} at a {THRESHOLD:.0%} cut-off, "
          f"catches {k5['recall']:.1%} of results that were not passes")
    print("By subject (5 tests known): " + ", ".join(f"{r['subject']} {r['roc_auc']:.3f}" for r in parts["by_subject_5_tests"]))
    print("By tests known: " + ", ".join(f"{r['tests_known']}: {r['roc_auc']:.3f}" for r in parts["by_tests_known"]))
    print("Next: python -m stud_ai.artifact_hashes; review candidate hashes and supply both STUD_AI_*_SHA256 pins before evaluation.")
    print("Evaluation updates the model card: review and promote its final hash after testing, before deployment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
