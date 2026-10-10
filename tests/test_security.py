import hashlib
import json
from unittest.mock import patch

import pytest
from catboost import CatBoostClassifier

from stud_ai.batch import CONTRACT, answer
from stud_ai.features import StudentRecord, features_for_record
from stud_ai.model import CARD_FILE, MODEL_FILE, IntegrityError, load_model
from stud_ai.predict import predict, predict_records


@pytest.mark.parametrize("target", ["model", "card"])
def test_tampered_artifacts_are_refused_before_deserialization(tmp_path, target):
    model, card = tmp_path / "copy.cbm", tmp_path / "card.json"
    model.write_bytes(MODEL_FILE.read_bytes())
    card.write_bytes(CARD_FILE.read_bytes())
    victim = model if target == "model" else card
    victim.write_bytes(victim.read_bytes() + b" ")
    with patch.object(CatBoostClassifier, "load_model") as loader:
        with pytest.raises(IntegrityError, match="integrity"):
            load_model(model, card)
        loader.assert_not_called()


def test_each_load_verifies_again_and_deserializes_the_verified_bytes(tmp_path):
    model, card = tmp_path / "copy.cbm", tmp_path / "card.json"
    model.write_bytes(MODEL_FILE.read_bytes())
    card.write_bytes(CARD_FILE.read_bytes())
    original = CatBoostClassifier.load_model
    def swap_file(self, *args, **kwargs):
        assert kwargs["blob"] == MODEL_FILE.read_bytes()
        model.write_bytes(b"changed after verification")
        return original(self, *args, **kwargs)
    with patch.object(CatBoostClassifier, "load_model", swap_file):
        assert load_model(model, card).model.tree_count_ == 239
    with pytest.raises(IntegrityError):
        load_model(model, card)


def test_production_requires_both_independent_pins(monkeypatch):
    monkeypatch.setenv("STUD_AI_PRODUCTION", "1")
    monkeypatch.delenv("STUD_AI_MODEL_SHA256", raising=False)
    monkeypatch.delenv("STUD_AI_CARD_SHA256", raising=False)
    with pytest.raises(IntegrityError):
        load_model()
    monkeypatch.setenv("STUD_AI_MODEL_SHA256", hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest())
    monkeypatch.setenv("STUD_AI_CARD_SHA256", hashlib.sha256(CARD_FILE.read_bytes()).hexdigest())
    assert load_model().model.tree_count_ > 0


@pytest.mark.parametrize("value", [None, 1, 1.0, True, {}, " " ])
def test_direct_python_categorical_types_are_strict(value):
    with pytest.raises(ValueError):
        features_for_record(StudentRecord(value, [45]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400, True, "45", None, -1, 101])
def test_direct_python_scores_are_strict(value):
    with pytest.raises(ValueError):
        features_for_record(StudentRecord("mathematics", [value]))


def test_unknown_categories_never_reach_the_model():
    loaded = load_model()
    with patch.object(loaded.model, "predict_proba") as spy:
        result = predict(StudentRecord("untrained subject", [45]), loaded=loaded)
    spy.assert_not_called()
    assert not result.known_subject and result.percent is None and result.risk is None
    assert result.text.startswith("Not enough data as of now") and result.reasons == []


def test_every_trained_subject_gets_its_percentage_but_no_raw_probability_leaves_the_batch():
    import numpy as np
    loaded = load_model()
    records = [StudentRecord("mathematics", [45])] * 5
    probabilities = np.array([[.9, .1], [.2, .8], [.5, .5], [.79, .21], [.8, .2]])
    with patch.object(loaded.model, "predict_proba", return_value=probabilities):
        result = predict_records(records, top_reasons=0, loaded=loaded)
    assert [r.percent for r in result] == [10, 80, 50, 21, 20]
    assert result[2].text == "Estimated risk of not passing the next mathematics examination: 50%"
    req = {"contract": CONTRACT, "students": [{"ref": "synthetic", "subjects": [
        {"key": "m", "name": "mathematics", "tests": [45], "attendance": None, "homework": None}]}]}
    output = answer(req, loaded)["results"][0]
    # The whole percent, its sentence and the reasons in words: never the raw probability, logits or SHAP weights.
    assert set(output) == {"ref", "key", "subject", "percent", "text", "reasons", "assumed", "known_subject", "tests_used"}
    assert isinstance(output["percent"], int) and output["reasons"]
    assert all(set(r) == {"key", "words", "value", "effect", "sentence"} for r in output["reasons"])
