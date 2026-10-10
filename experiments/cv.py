"""
Cross-validation harness for comparing model designs WITHOUT touching the test file.

Uses train_students.csv + validation_students.csv together (4,250 students), 5 folds by
student: all of a student's rows (every subject, 1..5 tests known) stay in one fold. Inside
each fold, 15% of the training students are held back for early stopping, so the scored
fold stays unseen.

    python experiments/cv.py                             # every design (about 4 minutes)
    python experiments/cv.py "one model, with subject"   # just one
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from stud_ai.features import FEATURES, NUMERIC, build_features, subjects_in  # noqa: E402
from stud_ai.model import MONOTONE, PARAMS, data_file  # noqa: E402

K_ALL = (1, 2, 3, 4, 5)
SUBJECTS = subjects_in(pd.read_csv(data_file("train"), nrows=0).columns)  # the subjects in the files, from their column names


def load() -> pd.DataFrame:
    return pd.concat([pd.read_csv(data_file("train")), pd.read_csv(data_file("validation"))], ignore_index=True)


def catboost_design(cols, mono=True, per_subject=False, **over):
    def fit(F: pd.DataFrame, seed: int):
        ids = F["student_id"].unique()
        fit_ids, stop_ids = train_test_split(ids, test_size=0.15, random_state=seed)
        a, b = F[F.student_id.isin(fit_ids)], F[F.student_id.isin(stop_ids)]
        params = {**PARAMS, **over, "random_seed": seed}
        if "subject" in cols:
            params["cat_features"] = ["subject"]
        if mono:
            params["monotone_constraints"] = [MONOTONE[c] for c in cols]
        m = CatBoostClassifier(**params)
        m.fit(a[cols], a["label_fail"], eval_set=(b[cols], b["label_fail"]), use_best_model=True)
        return m

    def fit_predict(Ftr, Fva, seed=42):
        if not per_subject:
            m = fit(Ftr, seed)
            return m.predict_proba(Fva[cols])[:, 1]
        p = np.zeros(len(Fva))
        for s in SUBJECTS:
            m = fit(Ftr[Ftr.subject == s], seed)
            mask = (Fva.subject == s).to_numpy()
            p[mask] = m.predict_proba(Fva.loc[mask, cols])[:, 1]
        return p

    return fit_predict


def logistic_design():
    def fit_predict(Ftr, Fva, seed=42):
        def X(F):
            out = F[list(NUMERIC)].copy()
            for s in SUBJECTS[1:]:
                out[f"is_{s}"] = (F["subject"] == s).astype(float)
            return out
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        m.fit(X(Ftr), Ftr["label_fail"])
        return m.predict_proba(X(Fva))[:, 1]
    return fit_predict


def cross_validate(df, fit_predict, folds=5, seed=7):
    rows = []
    for f, (tr, va) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(df)):
        Ftr = build_features(df.iloc[tr].reset_index(drop=True), K_ALL)
        Fva = build_features(df.iloc[va].reset_index(drop=True), K_ALL)
        p = fit_predict(Ftr, Fva)
        y = Fva["label_fail"].to_numpy()
        k5 = Fva["k"].to_numpy() == 5
        row = {"fold": f, "auc_all": roc_auc_score(y, p), "ll_all": log_loss(y, p), "brier_all": brier_score_loss(y, p),
               "auc_k5": roc_auc_score(y[k5], p[k5]), "acc_k5": float(((p[k5] >= 0.5) == (y[k5] == 1)).mean())}
        for s in SUBJECTS:
            m = k5 & (Fva["subject"].to_numpy() == s)
            row[f"auc_{s}"] = roc_auc_score(y[m], p[m])
            row[f"ll_{s}"] = log_loss(y[m], p[m])
        rows.append(row)
    return pd.DataFrame(rows)


ALL = list(FEATURES)
NO_SUBJECT = [c for c in FEATURES if c != "subject"]
DESIGNS = {
    "one model, with subject": lambda: catboost_design(ALL),
    "one model, without subject": lambda: catboost_design(NO_SUBJECT),
    "one model per subject": lambda: catboost_design(NO_SUBJECT, per_subject=True),
    "one model, without monotone rules": lambda: catboost_design(ALL, mono=False),
    "logistic regression (subject as 0/1 columns)": logistic_design,
}


def main(names=None):
    df = load()
    print(f"{len(df)} students x {len(SUBJECTS)} subjects; 5 folds by student\n")
    results = {}
    for name, factory in DESIGNS.items():
        if names and name not in names:
            continue
        t = time.time()
        r = cross_validate(df, factory())
        results[name] = r
        subj = " ".join(f"{s[:4]} {r[f'auc_{s}'].mean():.3f}" for s in SUBJECTS)
        print(f"{name:46s} AUC all {r.auc_all.mean():.4f} logloss {r.ll_all.mean():.4f} Brier {r.brier_all.mean():.4f} | "
              f"5 tests: AUC {r.auc_k5.mean():.4f} acc {r.acc_k5.mean():.4f} | {subj} | {time.time() - t:.0f}s", flush=True)
    base = results.get("one model, with subject")
    if base is not None:
        print("\nPaired differences vs 'one model, with subject' (negative log loss = better):")
        for name, r in results.items():
            if r is base:
                continue
            d = r.ll_all.to_numpy() - base.ll_all.to_numpy()
            da = r.auc_all.to_numpy() - base.auc_all.to_numpy()
            print(f"  {name:46s} ΔAUC {da.mean():+.4f}  Δlogloss {d.mean():+.4f}  folds better {(d < 0).sum()}/{len(d)}")
    return results


if __name__ == "__main__":
    main(sys.argv[1:] or None)
