import json
import re
from unittest.mock import patch

import pandas as pd
import pytest

from stud_ai.features import (
    FEATURES, MAX_TESTS, StudentRecord, SubjectMarks, build_features, feature_words, record_from_row, subject_words, subjects_in,
)
from stud_ai.model import CARD_FILE, MODEL_FILE, MONOTONE, data_file, load_model
from stud_ai.predict import predict, predict_frame, predict_student, risk_percent, sentence

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="train the model first: python -m stud_ai.train")

# The subjects in your dataset, read from the training file's column names (so these checks keep working
# after you retrain on the school's own records, whatever its subjects are).
DATA_SUBJECTS = subjects_in(pd.read_csv(data_file("train"), nrows=0).columns)
TEXT = re.compile(r"^Estimated risk of not passing the next (.+) examination: (\d{1,2})%$")


@pytest.fixture(scope="module")
def loaded():
    return load_model()


def base(subject=DATA_SUBJECTS[0], **changes) -> StudentRecord:
    values = dict(subject=subject, tests=[48, 52, 45], attendance=85, homework=70, other_average=55, other_homework=75, other_attendance=88)
    values.update(changes)
    return StudentRecord(**values)


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
def test_the_sentence_names_the_subject(loaded, subject):
    res = predict(base(subject), loaded=loaded)
    m = TEXT.match(res.text)
    assert m, res.text
    assert m.group(1) == subject_words(subject)
    assert 1 <= int(m.group(2)) <= 99 and int(m.group(2)) == res.percent
    assert 0 < res.risk < 1 and res.known_subject


def test_percent_is_never_0_or_100():
    assert risk_percent(0.0) == 1 and risk_percent(1.0) == 99 and risk_percent(0.284) == 28
    assert sentence("english", 28) == "Estimated risk of not passing the next English examination: 28%"
    assert sentence("Maths", 7) == "Estimated risk of not passing the next mathematics examination: 7%"
    assert sentence("social_science", 50) == "Estimated risk of not passing the next social science examination: 50%"


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
@pytest.mark.parametrize("change", [
    {"tests": [58, 62, 55]},
    {"homework": 95},
    {"attendance": 99},
    {"other_average": 75},
    {"other_homework": 95},
    {"other_attendance": 99},
])
def test_doing_better_never_raises_the_risk(loaded, subject, change):
    assert predict(base(subject, **change), loaded=loaded).risk <= predict(base(subject), loaded=loaded).risk + 1e-9


def test_doing_better_never_raises_the_risk_for_real_students(loaded):
    """The same rule on real rows: 60 students x every subject x 1-5 tests known, each input raised by 10 points."""
    X = build_features(pd.read_csv(data_file("validation")).head(60), range(1, MAX_TESTS + 1))[list(FEATURES)]
    before = loaded.model.predict_proba(X)[:, 1]
    for f in FEATURES:
        if MONOTONE[f] != -1:
            continue
        better = X.copy()
        better[f] = (better[f] + 10).clip(upper=100)
        after = loaded.model.predict_proba(better)[:, 1]
        assert (after <= before + 1e-9).all(), f"{f}: {(after > before + 1e-9).sum()} of {len(X)} rows got a higher risk"


def test_no_split_in_the_trees_sits_inside_rounding_noise(loaded):
    """A split at 0.00000000000001 would let invisible rounding differences change a student's risk."""
    names = loaded.model.feature_names_
    for i, borders in loaded.model.get_borders().items():
        assert not any(0 < abs(b) < 1e-7 for b in borders), names[i]


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
def test_much_better_marks_lower_the_risk_a_lot(loaded, subject):
    good = predict(base(subject, tests=[80, 85, 82]), loaded=loaded).risk
    weak = predict(base(subject, tests=[30, 28, 35]), loaded=loaded).risk
    assert good < weak - 0.5


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
def test_reasons_are_plain_sentences_about_the_subject(loaded, subject):
    res = predict(base(subject, tests=[30, 28, 35]), loaded=loaded)
    assert res.reasons and res.reasons[0].key == "scores" and res.reasons[0].effect == "raises"
    assert res.reasons[0].words == feature_words("scores", subject)
    for r in res.reasons:
        assert r.sentence.endswith(("(raises the risk)", "(lowers the risk)"))


def test_missing_inputs_are_filled_and_never_used_as_reasons(loaded):
    subject = DATA_SUBJECTS[-1]
    res = predict(StudentRecord(subject, [40, 42]), loaded=loaded)
    assert feature_words("attendance", subject) in res.assumed and "Average test score in other subjects" in res.assumed
    assert all(r.words not in res.assumed for r in res.reasons)


def test_a_subject_it_was_not_trained_on_has_not_enough_data(loaded):
    unseen = next(s for s in ("hindi", "sanskrit", "astronomy") if s not in DATA_SUBJECTS)
    res = predict(base(unseen), loaded=loaded)
    assert res.risk is None and res.percent is None and not res.known_subject
    assert res.text == f"Not enough data as of now for {subject_words(unseen)}" and res.reasons == [] and res.assumed == []


def test_csv_prediction_refuses_untrained_subject_before_catboost(loaded):
    df = pd.read_csv(data_file("test")).head(2)
    prefix = DATA_SUBJECTS[0].replace(" ", "_") + "_"
    df = df.rename(columns={c: "untrained_" + c[len(prefix):] for c in df.columns if c.startswith(prefix)})
    with patch.object(loaded.model, "predict_proba") as spy:
        with pytest.raises(ValueError, match="Not enough data as of now"):
            predict_frame(df, loaded=loaded)
    spy.assert_not_called()


def test_one_student_gets_a_sentence_for_every_subject(loaded):
    marks = {s: SubjectMarks([48, 52, 45], attendance=85, homework=70) for s in DATA_SUBJECTS}
    first = DATA_SUBJECTS[0]
    marks[first] = SubjectMarks([70, 75, 72], attendance=95, homework=90)
    results = predict_student(marks, loaded=loaded)
    assert [r.subject for r in results] == list(DATA_SUBJECTS)
    assert all(TEXT.match(r.text) for r in results)
    # The same as predicting one subject with the other subjects worked out by hand.
    last, n_others = DATA_SUBJECTS[-1], len(DATA_SUBJECTS) - 1
    others = {s: m.tests for s, m in marks.items() if s != last}
    by_hand = predict(StudentRecord(last, [48, 52, 45], 85, 70, other_tests=others,
                                    other_homework=(70 * (n_others - 1) + 90) / n_others,
                                    other_attendance=(85 * (n_others - 1) + 95) / n_others), loaded=loaded)
    assert results[-1].subject == last and results[-1].risk == pytest.approx(by_hand.risk)


def test_one_student_and_the_whole_file_agree(loaded):
    df = pd.read_csv(data_file("test")).head(20)
    batch = predict_frame(df, 3, None, loaded)
    assert list(batch["subject"].unique()) == list(DATA_SUBJECTS)
    for subject in DATA_SUBJECTS:
        rows = batch[batch.subject == subject].reset_index(drop=True)
        for i in range(0, 20, 7):
            one = predict(record_from_row(df.iloc[i], subject, 3), loaded=loaded)
            assert one.risk == pytest.approx(rows.iloc[i]["risk"], abs=1e-4)
            assert rows.iloc[i]["text"] == one.text


def test_the_model_card_matches_the_code_and_the_data(loaded):
    card = json.loads(CARD_FILE.read_text())
    assert card["features"] == list(FEATURES)
    assert card["subjects_trained_on"] == list(DATA_SUBJECTS)
    assert "gender" in " ".join(card["not_used"]) and "age" in card["not_used"]
    assert len(card["training_data"]["sha256"]) == 64
