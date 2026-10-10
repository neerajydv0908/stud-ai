"""Measures of how good the predicted risks are (shared by training and the test report)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def summary(y_fail: np.ndarray, risk: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """`y_fail` = 1 if the student did not pass; `risk` = predicted chance of not passing (0-1)."""
    y = np.asarray(y_fail, dtype=int)
    p = np.clip(np.asarray(risk, dtype=float), 1e-6, 1 - 1e-6)
    flag = p >= threshold
    tp = int(np.sum(flag & (y == 1)))
    fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1)))
    tn = int(np.sum(~flag & (y == 0)))
    both = len(np.unique(y)) == 2
    return {
        "students": int(len(y)),
        "actual_not_passed": float(y.mean()) if len(y) else float("nan"),
        "average_predicted_risk": float(p.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "pr_auc_not_passed": float(average_precision_score(y, p)) if both else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def calibration_table(y_fail: np.ndarray, risk: np.ndarray, edges=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    """For each band of predicted risk: how many students, their average predicted risk, and how many actually did not pass."""
    y = np.asarray(y_fail, dtype=float)
    p = np.asarray(risk, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({"band": f"{lo:.0%}–{min(hi, 1):.0%}", "students": int(m.sum()), "predicted": float(p[m].mean()), "actual": float(y[m].mean())})
    return rows


def expected_calibration_error(y_fail: np.ndarray, risk: np.ndarray) -> float:
    rows = calibration_table(y_fail, risk)
    n = sum(r["students"] for r in rows)
    return float(sum(r["students"] / n * abs(r["predicted"] - r["actual"]) for r in rows)) if n else float("nan")
