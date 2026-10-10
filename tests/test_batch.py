import json
import subprocess
import sys
import time

import pytest

from stud_ai.batch import CONTRACT, RequestError, answer, main
from stud_ai.features import StudentRecord
from stud_ai.model import MODEL_FILE, ROOT, load_model
from stud_ai.predict import predict

pytestmark = pytest.mark.skipif(not MODEL_FILE.exists(), reason="train the model first: python -m stud_ai.train")


@pytest.fixture(scope="module")
def loaded():
    return load_model()


def subject(key, name, tests, attendance=88.0, homework=75.0):
    return {"key": key, "name": name, "tests": tests, "attendance": attendance, "homework": homework}


def request(*students):
    return {"contract": CONTRACT, "students": [{"ref": ref, "subjects": list(subjects)} for ref, subjects in students]}


def by_key(response):
    return {(r["ref"], r["key"]): r for r in response["results"]}


def test_every_subject_gets_its_own_answer(loaded):
    res = answer(request(("s1", [subject("m", "Mathematics", [72, 64.5, 58]), subject("e", "English", [81, 77]),
                                 subject("h", "Hindi", [55], homework=None)])), loaded)
    got = by_key(res)
    assert set(got) == {("s1", "m"), ("s1", "e"), ("s1", "h")}
    hindi = got[("s1", "h")]
    assert hindi["known_subject"] is False and hindi["percent"] is None and hindi["reasons"] == [] and hindi["assumed"] == []
    assert hindi["text"] == "Not enough data as of now for Hindi"
    for r in res["results"]:
        assert set(r) == {"ref", "key", "subject", "percent", "text", "reasons", "assumed", "known_subject", "tests_used"}
        assert len(r["reasons"]) <= 3
        for reason in r["reasons"]:
            assert set(reason) == {"key", "words", "value", "effect", "sentence"} and reason["effect"] in {"raises", "lowers"}
        if r["known_subject"]:
            assert 1 <= r["percent"] <= 99 and r["text"].endswith(f" examination: {r['percent']}%")
    assert res["contract"] == CONTRACT and res["skipped"] == [] and res["errors"] == []
    assert res["model"]["trees"] > 0 and len(res["model"]["file_sha256"]) == 64 and res["model"]["subjects_trained_on"]


def test_the_answer_matches_predicting_one_subject_by_hand(loaded):
    res = answer(request(("s1", [subject("m", "Mathematics", [40, 45, 38], 72, 55), subject("e", "English", [62, 58], 72, 80),
                                 subject("x", "Science", [50], 72, None)])), loaded)
    maths = by_key(res)[("s1", "m")]
    by_hand = predict(StudentRecord("mathematics", [40, 45, 38], 72, 55, other_tests={"english": [62, 58], "science": [50]},
                                    other_homework=80, other_attendance=72), loaded=loaded)
    assert maths["percent"] == by_hand.percent and maths["text"] == by_hand.text
    assert [r["sentence"] for r in maths["reasons"]] == [r.sentence for r in by_hand.reasons]
    assert maths["assumed"] == by_hand.assumed


def test_subjects_without_tests_are_skipped_but_still_count_for_the_others(loaded):
    with_it = answer(request(("s1", [subject("m", "Mathematics", [50, 52]), subject("s", "Social Science", [], homework=20)])), loaded)
    without = answer(request(("s1", [subject("m", "Mathematics", [50, 52])])), loaded)
    assert with_it["skipped"] == [{"ref": "s1", "key": "s", "reason": "no test scores yet"}]
    assert [r["key"] for r in with_it["results"]] == ["m"]
    assert with_it["results"][0]["tests_used"] == without["results"][0]["tests_used"] == 2


def test_excess_test_history_is_refused_before_inference(loaded):
    res = answer(request(("s1", [subject("m", "Mathematics", [90, 30, 31, 29, 30, 28])])), loaded)
    assert not res["results"] and "at most 5" in res["errors"][0]["error"]


@pytest.mark.parametrize("bad, message", [
    (subject("x", "Science", [40, 140]), "from 0 to 100"),
    (subject("x", "Science", [10**400]), "from 0 to 100"),
    (subject("x", "Science", [float("nan")]), "from 0 to 100"),
    (subject("x", "Science", [True]), "must be a number"),
    (subject("x", "Science", [40, None]), "can't be null"),
    (subject("x", "Science", [40, "high"]), "must be a number"),
    (subject("x", "Science", [40], attendance=101), "attendance"),
    (subject("x", "Science", [40], homework=-1), "homework"),
    (subject("x", "Maths", [40]), "also called 'mathematics'"),
    ({"key": "x", "name": "", "tests": [40]}, "subject name"),
    ({"key": "x", "name": "Science", "tests": "40"}, "list"),
])
def test_a_bad_subject_is_refused_alone(loaded, bad, message):
    res = answer(request(("s1", [subject("m", "Mathematics", [50]), bad])), loaded)
    assert [r["key"] for r in res["results"]] == ["m"]
    assert len(res["errors"]) == 1 and res["errors"][0]["key"] == "x" and message in res["errors"][0]["error"]


@pytest.mark.parametrize("req, message", [
    ([], "JSON object"),
    ({"contract": 99, "students": []}, "Unknown contract"),
    ({"contract": True, "students": []}, "Unknown contract"),
    ({"contract": CONTRACT}, "must be a list"),
    (request(("s1", [])), None),  # fine: nothing to answer
    (request(("s1", []), ("s1", [])), "appears twice"),
    (request(("", [])), "ref"),
    (request(("s1", [subject("m", "Mathematics", [50]), subject("m", "English", [50])])), "appears twice"),
    (request(("s1", [subject(None, "Mathematics", [50])])), "key"),
])
def test_a_wrong_request_is_refused_as_a_whole(loaded, req, message):
    if message is None:
        assert answer(req, loaded)["results"] == []
        return
    with pytest.raises(RequestError, match=message):
        answer(req, loaded)


def test_an_answer_with_nothing_to_predict_still_names_the_verified_model():
    # Two names for one subject: one is refused as a duplicate, the other has no tests yet.
    res = answer(request(("s1", [subject("a", "Maths", [], homework=60), subject("b", "Mathematics", [55])])))
    assert res["results"] == [] and res["errors"]
    assert len(res["model"]["file_sha256"]) == 64 and len(res["model"]["card_sha256"]) == 64


def test_a_whole_school_is_quick(loaded):
    names = ["Hindi", "English", "Mathematics", "Science", "Social Science", "Sanskrit"]
    students = [(f"s{i}", [subject(f"k{j}", n, [30 + (i * 7 + j * 11) % 60, 35 + (i * 3 + j) % 55], 70 + i % 30, 40 + (i + j) % 60)
                           for j, n in enumerate(names)]) for i in range(500)]
    t = time.time()
    res = answer(request(*students), loaded)
    assert len(res["results"]) == 3000 and not res["errors"]
    assert time.time() - t < 30


def run_cli(stdin: str):
    return subprocess.run([sys.executable, "-m", "stud_ai.batch"], input=stdin.encode("utf-8"), capture_output=True, cwd=ROOT, timeout=120)


def test_the_command_line_speaks_json_in_utf8():
    req = request(("छात्र-1", [subject("m", "Mathematics", [44, 47]), subject("h", "हिंदी", [60])]))
    out = run_cli(json.dumps(req, ensure_ascii=False))
    assert out.returncode == 0, out.stderr.decode()
    res = json.loads(out.stdout.decode("utf-8"))
    assert {r["ref"] for r in res["results"]} == {"छात्र-1"}
    assert any("हिंदी" in r["subject"] for r in res["results"])


@pytest.mark.parametrize("stdin, code, message", [
    ("not json", 2, "not valid JSON"),
    ("", 2, "not valid JSON"),
    ('{"contract": 7, "students": []}', 2, "Unknown contract"),
])
def test_the_command_line_explains_a_wrong_request(stdin, code, message):
    out = run_cli(stdin)
    assert out.returncode == code and out.stdout == b""
    assert message in out.stderr.decode() and out.stderr.decode().count("\n") == 1


def test_files_in_and_out(tmp_path):
    (tmp_path / "in.json").write_text(json.dumps(request(("s1", [subject("m", "Mathematics", [50])]))), encoding="utf-8")
    assert main(["--in", str(tmp_path / "in.json"), "--out", str(tmp_path / "out.json")]) == 0
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["results"][0]["key"] == "m"


def test_cli_rejects_oversized_input():
    out = run_cli(" " * (2 * 1024 * 1024 + 1))
    assert out.returncode == 2 and b"exceeds" in out.stderr and b"Traceback" not in out.stderr


def test_cli_handles_arbitrarily_large_integer_without_traceback():
    out = run_cli(json.dumps(request(("s1", [subject("m", "Mathematics", [10**400])]))))
    assert out.returncode == 0 and b"Traceback" not in out.stderr
    value = json.loads(out.stdout)
    assert not value["results"] and value["errors"]
