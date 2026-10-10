"""
Score the test file ONCE with the trained model and write reports/test-report.md.

    python -m stud_ai.evaluate

The test students were never seen during training or while choosing settings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import Pool  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .features import CATEGORICAL, FEATURES, MAX_TESTS, NUMERIC, SCORE_FEATURES, build_features, record_from_row, subject_words  # noqa: E402
from .metrics import calibration_table, expected_calibration_error, summary  # noqa: E402
from .model import CARD_FILE, REPORT_DIR, THRESHOLD, data_file, load_model  # noqa: E402
from .predict import predict, predict_frame  # noqa: E402
from .train import TESTS_KNOWN, load_dataset, same_subjects  # noqa: E402

# Chart colours (one series): blue marks, dark text, muted secondary text, light grid.
BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"


def pct(x: float, digits: int = 1) -> str:
    return "–" if x != x else f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 3) -> str:
    return "–" if x != x else f"{x:.{digits}f}"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def calibration_chart(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.plot([0, 100], [0, 100], linestyle=(0, (4, 4)), color="#9a988f", linewidth=1.2)
    ax.text(98, 6, "Dashed line: stated risk = what happened", color=MUTED, fontsize=8.5, ha="right")
    x = [r["predicted"] * 100 for r in rows]
    y = [r["actual"] * 100 for r in rows]
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for xi, yi, r in zip(x, y, rows):
        ax.annotate(f"{r['students']}", (xi, yi), textcoords="offset points", xytext=(6, -12), fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Predicted risk of not passing (%)", color=INK, fontsize=9.5)
    ax.set_ylabel("Results that were actually not a pass (%)", color=INK, fontsize=9.5)
    ax.set_title("Predicted risk vs what happened — test set, all subjects, 5 tests known", color=INK, fontsize=10.5, loc="left")
    fig.text(0.01, 0.01, "Each dot is a 10%-wide band of predicted risk; the small number is how many subject results fell in it.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def importance_chart(items: list[tuple[str, float]], path: Path) -> None:
    items = sorted(items, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(items) + 1.2), dpi=160, facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    labels = [t[0] for t in items]
    values = [t[1] * 100 for t in items]
    ax.barh(labels, values, color=BLUE, height=0.55)
    for i, v in enumerate(values):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(axis="y", colors=INK, labelsize=9)
    ax.set_xlabel("Share of the model's reasoning (average size of each input's effect)", color=INK, fontsize=9)
    # Title across the whole figure (not just above the bars), so a long title is never cut off.
    fig.suptitle("What moves the estimate — test set, all subjects, 5 tests known", color=INK, fontsize=10.5, x=0.015, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def grouped_importance(model, X: pd.DataFrame) -> list[tuple[str, float]]:
    """Each input's share of the model's reasoning: the average size of its SHAP effect (the subject's own scores counted together)."""
    signed = model.get_feature_importance(data=Pool(X, cat_features=list(CATEGORICAL)), type="ShapValues")[:, :-1]
    idx = {f: j for j, f in enumerate(FEATURES)}
    groups = {
        "The subject's own test scores so far": [idx[f] for f in SCORE_FEATURES],
        "Average test score in other subjects": [idx["other_average"]],
        "Which subject it is": [idx["subject"]],
        "Homework done in the subject": [idx["homework"]],
        "Homework done in other subjects": [idx["other_homework"]],
        "Attendance in the subject": [idx["attendance"]],
        "Attendance in other subjects": [idx["other_attendance"]],
    }
    raw = {name: float(np.mean(np.abs(signed[:, cols].sum(axis=1)))) for name, cols in groups.items()}
    total = sum(raw.values())
    return [(name, v / total) for name, v in raw.items()]


def _design(F: pd.DataFrame, cols: list[str], subjects) -> pd.DataFrame:
    out = F[cols].copy()
    for s in subjects[1:]:
        out[f"is_{s}"] = (F["subject"] == s).astype(float)
    return out


def baselines(train_df: pd.DataFrame, test: pd.DataFrame, subjects) -> list[tuple[str, dict]]:
    """Simpler models for comparison, trained on the same training rows."""
    tr = build_features(train_df, TESTS_KNOWN, subjects)
    k5 = test["k"] == MAX_TESTS
    out = []
    for name, cols in (("Logistic regression, same inputs", list(NUMERIC)), ("The subject's own average only (logistic regression)", ["average", "tests_known"])):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(_design(tr, cols, subjects), tr["label_fail"])
        out.append((name, summary(test.loc[k5, "label_fail"].to_numpy(), m.predict_proba(_design(test.loc[k5], cols, subjects))[:, 1], THRESHOLD)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the test set once and write reports/test-report.md")
    ap.add_argument("--data", type=Path, help="folder with the CSV files (default: data/ when it holds them, else the project folder)")
    args = ap.parse_args(argv)

    loaded = load_model()
    model, card = loaded.model, loaded.card
    test_path = data_file("test", args.data)
    test_df = load_dataset(test_path)
    train_df = load_dataset(data_file("train", args.data))
    val_df = load_dataset(data_file("validation", args.data))
    leaked = (set(train_df["student_id"]) | set(val_df["student_id"])) & set(test_df["student_id"])
    if leaked:
        raise ValueError(f"{len(leaked)} test students also appear in the training or validation files.")
    subjects = same_subjects(("the training file", train_df), ("the validation file", val_df), (test_path.name, test_df))
    if set(card.get("subjects_trained_on", [])) != set(subjects):
        raise ValueError(f"The saved model was trained on {card.get('subjects_trained_on')}, but the files have {list(subjects)}. Retrain it.")

    test = build_features(test_df, TESTS_KNOWN, subjects)
    test["risk"] = model.predict_proba(test[list(FEATURES)])[:, 1]
    y = test["label_fail"].to_numpy()
    risk = test["risk"].to_numpy()
    k5 = (test["k"] == MAX_TESTS).to_numpy()
    s5 = summary(y[k5], risk[k5], THRESHOLD)
    by_k = [(k, summary(y[test["k"].to_numpy() == k], risk[test["k"].to_numpy() == k], THRESHOLD)) for k in TESTS_KNOWN]
    by_subject = []
    for subj in subjects:
        m = k5 & (test["subject"].to_numpy() == subj)
        by_subject.append((subj, summary(y[m], risk[m], THRESHOLD)))
    cal = calibration_table(y[k5], risk[k5])
    ece = expected_calibration_error(y[k5], risk[k5])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    calibration_chart(cal, REPORT_DIR / "calibration.png")
    importance_chart(grouped_importance(model, test.loc[k5, list(FEATURES)]), REPORT_DIR / "importance.png")

    # Fairness check, if the test file says the students' gender and age (the model never sees them).
    groups = test_df[[c for c in ("student_id", "gender", "age") if c in test_df.columns]].copy()
    if "age" in groups:
        groups["age_group"] = pd.cut(groups["age"], bins=[0, 16, 18, 200], labels=["15–16", "17–18", "19 and over"])
    t5 = test[k5].merge(groups, on="student_id")
    fairness = []
    for col in [c for c in ("gender", "age_group") if c in t5.columns]:
        for value, part in t5.groupby(col, observed=True):
            fairness.append((col, str(value), summary(part["label_fail"].to_numpy(), part["risk"].to_numpy(), THRESHOLD)))

    thresholds = [(t, summary(y[k5], risk[k5], t)) for t in (0.3, 0.4, 0.5, 0.6, 0.7)]
    base = baselines(train_df, test, subjects)

    # Three examples with 3 tests known: low, middle and high risk, in (up to) three different subjects.
    examples = []
    for i, q in enumerate((0.2, 0.8, 0.97)):
        subj = subjects[i % len(subjects)]
        preds = predict_frame(test_df, 3, [subj], loaded).sort_values("risk").reset_index(drop=True)
        sid = preds.iloc[int(len(preds) * q)]["student_id"]
        row = test_df[test_df["student_id"] == sid].iloc[0]
        examples.append((sid, subj, row, predict(record_from_row(row, subj, 3), loaded=loaded)))

    predict_frame(test_df, MAX_TESTS, subjects, loaded).to_csv(REPORT_DIR / "test-predictions.csv", index=False)

    n_res = int(k5.sum())
    L = ["# Stud-AI risk model — test report", ""]
    L += [f"Model: CatBoost, one model for every subject, {card.get('trees', '?')} trees, trained {card.get('created', '?')[:10]} on "
          f"`{card.get('training_data', {}).get('file', '?')}` ({card.get('training_data', {}).get('students', '?')} students), settings checked and "
          f"training stopped on `{card.get('validation_data', {}).get('file', '?')}` ({card.get('validation_data', {}).get('students', '?')} students). "
          f"Scored here on `{test_path.name}`: **{len(test_df)} students the model never saw**, in {len(subjects)} subjects "
          f"({n_res} subject results, {y[k5].mean():.1%} of them not a pass).", ""]
    L += ["**Question answered:** how likely is it that this student will *not* pass the next examination in a subject?  ",
          "**Shown as:** *Estimated risk of not passing the next science examination: 28%*", ""]
    L += ["**Inputs:** the subject; in that subject, attendance %, homework done % and the test scores so far (latest, average, lowest, "
          "highest, trend, how many); and the other subjects' average test score, homework % and attendance %.  ",
          "**Not used:** name, ID, gender, age, and the other subjects' pass/fail (decided at the same time, so unknown beforehand).", ""]

    L += ["## Headline (all subjects, all 5 tests known)", "", "| Measure | Result | What it means |", "| --- | ---: | --- |"]
    L += [f"| ROC-AUC | {num(s5['roc_auc'])} | Ranks a result that was not a pass above one that was {pct(s5['roc_auc'], 0)} of the time (0.5 = guessing, 1 = perfect). |"]
    L += [f"| Accuracy at {THRESHOLD:.0%} | {pct(s5['accuracy'])} | Results on the right side of the {THRESHOLD:.0%} line. |"]
    L += [f"| Not-passes caught | {pct(s5['recall'])} | {s5['tp']} of {s5['tp'] + s5['fn']} had a risk of {THRESHOLD:.0%} or more. |"]
    L += [f"| Right when it says {THRESHOLD:.0%}+ | {pct(s5['precision'])} | {s5['tp']} of {s5['tp'] + s5['fp']} flagged results were not passes. |"]
    L += [f"| Brier score | {num(s5['brier'])} | Average squared error of the risk (lower is better; always guessing {pct(y[k5].mean(), 0)} would score {num(y[k5].mean() * (1 - y[k5].mean()))}). |"]
    L += [f"| Calibration error | {pct(ece)} | On average the stated risk is this far from what actually happened. |", ""]

    L += ["## By subject (5 tests known)", "",
          "| Subject | Students | Did not pass | ROC-AUC | Accuracy at 50% | Caught (did not pass) | Right when flagged | Brier |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for subj, s in by_subject:
        name = subject_words(subj)
        L += [f"| {name[0].upper() + name[1:]} | {s['students']} | {pct(s['actual_not_passed'])} | {num(s['roc_auc'])} | {pct(s['accuracy'])} | "
              f"{pct(s['recall'])} | {pct(s['precision'])} | {num(s['brier'])} |"]
    L += [""]

    L += ["## With fewer tests known", "",
          "Early in the year there are fewer test scores, so the model is less sure. Each test student was scored using only their first 1, 2, … 5 tests in each subject.", "",
          "| Tests known | ROC-AUC | Accuracy at 50% | Caught (did not pass) | Right when flagged | Brier |", "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, s in by_k:
        L += [f"| {k} | {num(s['roc_auc'])} | {pct(s['accuracy'])} | {pct(s['recall'])} | {pct(s['precision'])} | {num(s['brier'])} |"]
    L += [""]

    L += ["## Where to draw the line", "",
          "A lower cut-off catches more students who will struggle, at the cost of more false alarms (all subjects, 5 tests known).", "",
          "| Risk cut-off | Results flagged | Caught (did not pass) | Right when flagged | Accuracy |", "| ---: | ---: | ---: | ---: | ---: |"]
    for t, s in thresholds:
        L += [f"| {t:.0%} | {pct((s['tp'] + s['fp']) / s['students'])} | {pct(s['recall'])} | {pct(s['precision'])} | {pct(s['accuracy'])} |"]
    L += [""]

    L += ["## Does a stated risk mean what it says?", "", "![Predicted risk against what happened](calibration.png)", "",
          "| Predicted risk | Subject results | Average predicted | Actually not a pass |", "| --- | ---: | ---: | ---: |"]
    for r in cal:
        L += [f"| {r['band']} | {r['students']} | {pct(r['predicted'])} | {pct(r['actual'])} |"]
    L += [""]

    L += ["## Same quality for every group?", ""]
    if fairness:
        L += ["The model is never told a student's gender or age. These rows check that its errors don't fall more on one group (all subjects, 5 tests known).", "",
              "| Group | Subject results | Not a pass | Average predicted risk | ROC-AUC | Caught (did not pass) | False alarms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for col, value, s in fairness:
            L += [f"| {'Gender: ' if col == 'gender' else 'Age: '}{value} | {s['students']} | {pct(s['actual_not_passed'])} | {pct(s['average_predicted_risk'])} | "
                  f"{num(s['roc_auc'])} | {pct(s['recall'])} | {pct(s['false_positive_rate'])} |"]
        L += ["", "*False alarms* = share of passes that were flagged at 50% or more. Gaps of a few points can be chance; check again on real data.", ""]
    else:
        L += ["Not checked: the test file has no `gender` or `age` column. Add them to the test file (the model still won't read them) to run this check.", ""]

    L += ["## What moves the estimate", "", "![What moves the estimate](importance.png)", ""]

    L += ["## Compared with simpler models (all subjects, 5 tests known, same training data)", "",
          "| Model | ROC-AUC | Brier | Accuracy at 50% |", "| --- | ---: | ---: | ---: |",
          f"| **CatBoost (this model)** | {num(s5['roc_auc'])} | {num(s5['brier'])} | {pct(s5['accuracy'])} |"]
    for name, s in base:
        L += [f"| {name} | {num(s['roc_auc'])} | {num(s['brier'])} | {pct(s['accuracy'])} |"]
    L += [""]

    L += ["## Three test students, 3 tests known", ""]
    for sid, subj, row, res in examples:
        status = "did not pass" if int(row[f"{subj}_passed"]) == 0 else "passed"
        L += [f"**{sid}, {subject_words(subj)}** — tests {', '.join(str(int(row[f'{subj}_test_{i}_score'])) for i in range(1, 4))}; "
              f"attendance {row[f'{subj}_attendance_percent']:.0f}%, homework {row[f'{subj}_homework_completion_percent']:.0f}% (actually {status})", "",
              f"> {res.text}"]
        L += [f"> - {r.sentence}" for r in res.reasons]
        L += [""]

    L += ["## Before using it on real students", "",
          "- This model learned from the dataset you supplied. Before teachers rely on it, retrain and re-test it on AESRI's own past records "
          "(`python -m stud_ai.train` then `python -m stud_ai.evaluate`), because real schools differ.",
          "- Subjects it wasn't trained on (Hindi, Sanskrit, EVS, …) still get an estimate from the same inputs, but a less reliable one, until they are in the training data.",
          "- Re-check it every year, and whenever the exams or marking change.",
          "- An estimate is a prompt to help a student, not a judgement: show it to teachers only, with its reasons.", ""]
    L += ["Files: `test-predictions.csv` (every test student and subject, 5 tests known), `calibration.png`, `importance.png`, "
          "`validation-search.md` (settings tried), `improvement-experiments.md` (designs compared)."]

    (REPORT_DIR / "test-report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    card["test"] = {"file": test_path.name, "students": len(test_df), "all_subjects_5_tests": s5,
                    "by_subject_5_tests": [{"subject": sj, **s} for sj, s in by_subject],
                    "by_tests_known": [{"tests_known": k, **s} for k, s in by_k], "calibration_error": ece}
    CARD_FILE.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"Test set, all subjects, 5 tests known: ROC-AUC {s5['roc_auc']:.3f}, accuracy {s5['accuracy']:.1%}, caught {s5['recall']:.1%} of not-passes, "
          f"right {s5['precision']:.1%} of the time when flagged, Brier {s5['brier']:.3f}, calibration error {ece:.1%}")
    print("By subject: " + ", ".join(f"{sj} {s['roc_auc']:.3f}" for sj, s in by_subject))
    print("By tests known: " + ", ".join(f"{k}: {s['roc_auc']:.3f}" for k, s in by_k))
    print(f"Wrote {REPORT_DIR / 'test-report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
