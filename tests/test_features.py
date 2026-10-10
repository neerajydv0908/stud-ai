import math

import numpy as np
import pandas as pd
import pytest

from stud_ai.features import (
    FEATURES, MAX_TESTS, StudentRecord, SubjectMarks, build_features, column_prefix, features_for_record, label_column,
    normalise_subject, record_from_row, records_for_student, subject_words, subjects_in,
)
from stud_ai.features import test_columns as score_columns  # renamed so pytest doesn't collect it as a test
from stud_ai.model import data_file

# The subjects in your dataset, read from the training file's column names.
DATA_SUBJECTS = subjects_in(pd.read_csv(data_file("train"), nrows=0).columns)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(data_file("validation")).head(30)


def tiny_school(subjects, n=12, seed=0) -> pd.DataFrame:
    """A made-up file in the dataset's format, with any subjects."""
    rng = np.random.default_rng(seed)
    data = {"student_id": [f"S{i:03d}" for i in range(n)]}
    for s in subjects:
        p = column_prefix(s)
        data[f"{p}_attendance_percent"] = rng.integers(60, 101, n)
        data[f"{p}_homework_completion_percent"] = rng.integers(30, 101, n)
        for i in range(1, MAX_TESTS + 1):
            data[f"{p}_test_{i}_score"] = rng.integers(15, 96, n)
        data[f"{p}_passed"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_the_subjects_come_from_the_column_names(sample):
    assert len(DATA_SUBJECTS) >= 2
    assert subjects_in(sample.columns) == DATA_SUBJECTS  # the validation file has the same subjects as the training file


def test_one_row_per_student_subject_and_number_of_tests(sample):
    F = build_features(sample, range(1, MAX_TESTS + 1))
    assert len(F) == len(sample) * len(DATA_SUBJECTS) * MAX_TESTS
    assert [c for c in F.columns if c in FEATURES] == list(FEATURES)
    assert set(F["subject"]) == set(DATA_SUBJECTS)
    assert set(F["k"]) == set(range(1, MAX_TESTS + 1))
    assert set(F["label_fail"]) <= {0, 1}


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
def test_labels_and_scores_belong_to_the_subject(sample, subject):
    F = build_features(sample, (5,), [subject])
    tests = sample[score_columns(subject)].to_numpy(float)
    assert np.allclose(F["average"], tests.mean(axis=1))
    assert np.allclose(F["latest"], tests[:, -1])
    assert (F["label_fail"].to_numpy() == 1 - sample[label_column(subject)].to_numpy()).all()
    others = [s for s in DATA_SUBJECTS if s != subject]
    expected = np.mean([sample[score_columns(o)].to_numpy(float).mean(axis=1) for o in others], axis=0)
    assert np.allclose(F["other_average"], expected)


def test_with_one_test_everything_is_that_test(sample):
    subject = DATA_SUBJECTS[-1]
    F = build_features(sample, (1,), [subject])
    first = sample[score_columns(subject)[0]].to_numpy(float)
    for col in ("latest", "average", "lowest", "highest"):
        assert np.allclose(F[col], first)
    assert (F["trend"] == 0).all()


def test_rising_scores_have_a_positive_trend():
    assert features_for_record(StudentRecord("english", [40, 50, 60]))["trend"] == pytest.approx(10)
    assert features_for_record(StudentRecord("english", [70, 60]))["trend"] == pytest.approx(-10)


def test_flat_scores_have_a_trend_of_exactly_zero():
    """Not 0.00000000000003: rounding noise like that used to decide splits in the trees."""
    for scores in ([50, 50], [37, 37, 37], [61, 61, 61, 61], [73, 73, 73, 73, 73], [50, 55, 50]):
        assert features_for_record(StudentRecord("english", scores))["trend"] == 0.0
    df = tiny_school(("mathematics", "english"))
    for i in range(1, MAX_TESTS + 1):
        df[f"english_test_{i}_score"] = 44
    F = build_features(df, range(1, MAX_TESTS + 1), ["english"])
    assert (F["trend"] == 0.0).all()


def test_every_input_is_rounded_the_same_way():
    f = features_for_record(StudentRecord("english", [50, 51, 51], other_tests={"science": [40, 41, 41]}))
    assert f["average"] == 50.666667 and f["other_average"] == 40.666667


@pytest.mark.parametrize("subject", DATA_SUBJECTS)
@pytest.mark.parametrize("k", [1, 3, 5])
def test_single_student_matches_the_batch(sample, subject, k):
    """The portal will send one student at a time; it must get exactly what training saw."""
    F = build_features(sample, (k,), [subject])
    for i in range(0, len(sample), 7):
        one = features_for_record(record_from_row(sample.iloc[i], subject, k))
        for name in FEATURES:
            if name == "subject":
                assert one[name] == F.iloc[i][name]
            else:
                assert one[name] == pytest.approx(F.iloc[i][name], abs=1e-9), name


def test_a_school_with_other_subjects_needs_no_code_change():
    """E.g. AESRI's own records, with social science and Hindi."""
    subjects = ("mathematics", "social science", "hindi", "environmental studies")
    df = tiny_school(subjects)
    assert "social_science_test_1_score" in df.columns
    assert subjects_in(df.columns) == subjects
    F = build_features(df, (1, 5))
    assert set(F["subject"]) == set(subjects) and len(F) == len(df) * len(subjects) * 2
    maths = F[(F.subject == "mathematics") & (F.k == 5)]
    others = [score_columns(s) for s in subjects[1:]]
    assert np.allclose(maths["other_average"], np.mean([df[c].to_numpy(float).mean(axis=1) for c in others], axis=0))
    social = F[(F.subject == "social science") & (F.k == 5)].reset_index(drop=True)
    for i in (0, 5):
        one = features_for_record(record_from_row(df.iloc[i], "Social_Science", 5))
        assert one["subject"] == "social science"
        for name in FEATURES[1:]:
            assert one[name] == pytest.approx(social.iloc[i][name]), name


def test_badly_named_subject_columns_are_explained():
    with pytest.raises(ValueError, match="mathematics_test_1_score"):
        subjects_in(["student_id", "Maths_test_1_score"])
    with pytest.raises(ValueError, match="No subject columns"):
        build_features(pd.DataFrame({"student_id": [1], "marks": [50]}))


def test_subject_names():
    assert normalise_subject("  Maths ") == "mathematics"
    assert normalise_subject("Social_Science") == "social science"
    assert normalise_subject("EVS") == "environmental studies"
    assert column_prefix("Social Science") == "social_science"
    assert subject_words("english") == "English"
    assert subject_words("hindi") == "Hindi"
    assert subject_words("mathematics") == "mathematics"


def test_the_subject_itself_is_never_counted_as_another_subject():
    f = features_for_record(StudentRecord("science", [50], other_tests={"science": [90], "english": [60]}))
    assert f["other_average"] == 60


def test_missing_values_stay_missing_for_the_predictor_to_fill():
    f = features_for_record(StudentRecord("history", [50]))
    assert math.isnan(f["attendance"]) and math.isnan(f["other_average"])


def test_one_student_all_subjects_works_out_the_other_subjects():
    recs = records_for_student({
        "Maths": SubjectMarks([40, 50], attendance=80, homework=60),
        "english": SubjectMarks([70], attendance=90),
        "science": SubjectMarks([60, 64], homework=80),
    })
    assert [r.subject for r in recs] == ["mathematics", "english", "science"]
    maths = features_for_record(recs[0])
    assert maths["average"] == 45 and maths["other_average"] == pytest.approx((70 + 62) / 2)
    assert maths["other_homework"] == 80 and maths["other_attendance"] == 90  # only the subjects that have them
    english = features_for_record(recs[1])
    assert math.isnan(english["homework"]) and english["other_homework"] == pytest.approx(70)


@pytest.mark.parametrize(
    "record, message",
    [
        (StudentRecord("science", []), "between 1 and 5"),
        (StudentRecord("science", [50] * 6), "between 1 and 5"),
        (StudentRecord("science", [120]), "Science test scores must be a percentage between 0 and 100"),
        (StudentRecord("science", [50], attendance=140), "percentage"),
        (StudentRecord("science", [50], other_tests={"english": [101]}), "English test scores"),
        (StudentRecord(" ", [50]), "subject"),
    ],
)
def test_bad_input_is_refused(record, message):
    with pytest.raises(ValueError, match=message):
        features_for_record(record)


@pytest.mark.parametrize(
    "subjects, message",
    [
        ({"maths": SubjectMarks([50]), "Mathematics": SubjectMarks([60])}, "twice"),
        ({"maths": SubjectMarks([50], homework=120), "english": SubjectMarks([60])}, "Mathematics homework"),
    ],
)
def test_bad_student_input_is_refused(subjects, message):
    with pytest.raises(ValueError, match=message):
        records_for_student(subjects)
