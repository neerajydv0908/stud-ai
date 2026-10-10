"""Where the model and its card live, the training settings, and loading a saved model."""
from __future__ import annotations

import json
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostClassifier

from .features import CATEGORICAL, FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "pass_risk.cbm"
CARD_FILE = MODEL_DIR / "model_card.json"
REPORT_DIR = ROOT / "reports"

# The dataset files. AESRI's own records go in data/, which git never publishes; when data/ holds any of
# them, all three are read from there. Otherwise the dataset you supplied, in the project folder, is used.
DATA_FILES = {"train": "train_students.csv", "validation": "validation_students.csv", "test": "test_students.csv"}
OWN_DATA_DIR = "data"

# Higher values of these can only lower the predicted risk (never raise it), all else being equal:
# better marks, more homework or better attendance must never make a student look more at risk.
# -1 = risk goes down as the value goes up; 0 = no rule.
MONOTONE: dict[str, int] = {f: -1 for f in FEATURES}
for _f in ("subject", "tests_known", "trend"):
    MONOTONE[_f] = 0

# Chosen with cross-validation and the validation set (see reports/improvement-experiments.md).
PARAMS = dict(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=30,
    # The subject is read as plain yes/no columns (is it mathematics? is it English? ...). Without this, CatBoost
    # mixes the subject with the numbers into combined "CTR" features, and those quietly break the rules above:
    # in testing, about 1 in 3 students got a HIGHER risk after their attendance went up. With it, no rule is
    # broken and the results are the same (validation log loss 0.3479 against 0.3480).
    one_hot_max_size=10,
    random_seed=42,
    od_type="Iter",
    od_wait=300,
    allow_writing_files=False,
    verbose=False,
)

THRESHOLD = 0.5  # a risk of 50% or more counts as "likely not to pass" in the test report


def data_folder(data_dir: Path | None = None) -> Path:
    """Where the dataset files are read from: the folder asked for; else data/ when it holds any of them;
    else the project folder. All three always come from the same folder, so a model is never trained on
    one dataset and tested on another."""
    if data_dir:
        return Path(data_dir)
    own = ROOT / OWN_DATA_DIR
    if any((own / filename).is_file() for filename in DATA_FILES.values()):
        return own
    return ROOT


def data_file(name: str, data_dir: Path | None = None) -> Path:
    """The path of one dataset file (train / validation / test)."""
    filename = DATA_FILES[name]
    folder = data_folder(data_dir)
    path = folder / filename
    if path.is_file():
        return path
    if not data_dir and folder != ROOT:
        raise FileNotFoundError(
            f"{filename} not found in {folder}. That folder holds some of the dataset files, so all three are read "
            f"from it: put {', '.join(DATA_FILES.values())} there together."
        )
    raise FileNotFoundError(f"{filename} not found in {folder}.")


def new_model(**overrides) -> CatBoostClassifier:
    params = {**PARAMS, **overrides}
    return CatBoostClassifier(**params, cat_features=list(CATEGORICAL), monotone_constraints=[MONOTONE[f] for f in FEATURES])


@dataclass
class LoadedModel:
    model: CatBoostClassifier
    card: dict
    model_sha256: str = ""
    card_sha256: str = ""


class IntegrityError(RuntimeError):
    """An artifact does not match the independently trusted release pins."""


def artifact_pins() -> tuple[str, str]:
    """Production pins come from deployment secrets; development pins ship with code.

    Never learn trust from the models directory or automatically repin on load.
    """
    model = os.environ.get("STUD_AI_MODEL_SHA256", "")
    card = os.environ.get("STUD_AI_CARD_SHA256", "")
    production = os.environ.get("NODE_ENV") == "production" or os.environ.get("STUD_AI_PRODUCTION") == "1"
    if not model and not card and not production:
        try:
            trusted = json.loads(Path(__file__).with_name("trusted_artifacts.json").read_text())
            model, card = trusted["model_sha256"], trusted["card_sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            raise IntegrityError("Trusted model and card SHA-256 pins are required.") from None
    if not all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in (model, card)):
        raise IntegrityError("Valid STUD_AI_MODEL_SHA256 and STUD_AI_CARD_SHA256 pins are required.")
    return model, card


def load_model(model_file: Path = MODEL_FILE, card_file: Path = CARD_FILE) -> LoadedModel:
    if not model_file.exists():
        raise FileNotFoundError(f"No trained model at {model_file}. Run: python -m stud_ai.train")
    expected_model, expected_card = artifact_pins()
    weights, card_bytes = model_file.read_bytes(), card_file.read_bytes()
    model_sha, card_sha = hashlib.sha256(weights).hexdigest(), hashlib.sha256(card_bytes).hexdigest()
    if not hmac.compare_digest(model_sha, expected_model) or not hmac.compare_digest(card_sha, expected_card):
        raise IntegrityError("Stud-AI artifact integrity verification failed; inference was refused.")
    card = json.loads(card_bytes)
    model = CatBoostClassifier()
    # Deserialize precisely the verified bytes, not a path that can change after hashing.
    model.load_model(blob=weights, format="cbm")
    names = list(model.feature_names_ or [])
    if names and tuple(names) != FEATURES:
        raise RuntimeError(f"The saved model expects {names}, but this code builds {list(FEATURES)}. Retrain it.")
    if card.get("features") != list(FEATURES) or card.get("trees") != model.tree_count_:
        raise IntegrityError("The verified model and card do not describe the same model.")
    subjects = card.get("subjects_trained_on")
    if not isinstance(subjects, list) or not subjects or any(not isinstance(s, str) or not s for s in subjects):
        raise IntegrityError("The model card must list its trained subjects.")
    return LoadedModel(model=model, card=card, model_sha256=model_sha, card_sha256=card_sha)
