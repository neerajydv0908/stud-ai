"""
Stud-AI for the Campus Portal (contract 3): many students at once, JSON in and JSON out.

    python -m stud_ai.batch < request.json > response.json
    python -m stud_ai.batch --in request.json --out response.json

The portal sends only numbers, under its own references (no names, nothing else about the child):

    {"contract": 3,
     "students": [
       {"ref": "8f1c...", "subjects": [
         {"key": "c27a...", "name": "Mathematics", "tests": [72, 64.5], "attendance": 88.5, "homework": 75},
         {"key": "91be...", "name": "Hindi", "tests": [55], "attendance": 88.5, "homework": null}]}]}

- tests: that subject's test scores so far, as percentages (marks / maximum marks x 100), oldest first,
  at most 5 (the latest ones).
- attendance, homework: percentages, or null when not known. A typical value is then used, and the
  answer lists it under "assumed" (it is never given as a reason).
- A subject with no test scores yet gets no estimate ("skipped"), but its homework and attendance still
  count towards the student's other subjects.
- Requests are limited to 2 MiB, 500 students, 40 subjects each and five scores per subject.

The answer, for every subject: the estimated risk of not passing the next examination as a whole
percent, its sentence, the main reasons and the inputs that were assumed. A subject the model wasn't
trained on gets no estimate (percent null, "Not enough data as of now", known_subject false) and never
reaches the model. Raw probabilities, logits and SHAP weights are never part of the answer.

    {"contract": 3,
     "model": {"version": "1.1.0", "created": "...", "trees": 239, "subjects_trained_on": [...],
               "file_sha256": "...", "card_sha256": "..."},
     "results": [{"ref": "8f1c...", "key": "c27a...", "subject": "mathematics", "percent": 28,
                  "text": "Estimated risk of not passing the next mathematics examination: 28%",
                  "reasons": [{"key": "scores", "words": "...", "value": "...", "effect": "raises", "sentence": "..."}],
                  "assumed": ["Homework done in mathematics"], "known_subject": true, "tests_used": 2},
                 {"ref": "8f1c...", "key": "91be...", "subject": "hindi", "percent": null,
                  "text": "Not enough data as of now for Hindi", "reasons": [], "assumed": [],
                  "known_subject": false, "tests_used": 1}],
     "skipped": [{"ref": "...", "key": "...", "reason": "no test scores yet"}],
     "errors": [{"ref": "...", "key": "...", "error": "..."}]}

Exit codes: 0 = answered (single subjects that were refused are listed under "errors"),
2 = the request itself is wrong, 3 = no trained model, 4 = the model or its card failed the integrity
check (or another configuration problem). Problems are explained in one line on stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features import MAX_TESTS, SubjectMarks, normalise_subject, records_for_student
from .model import MODEL_FILE, LoadedModel, load_model, IntegrityError
from .predict import RiskResult, predict_records

CONTRACT = 3
TOP_REASONS = 3  # the main reasons given with each estimate
MAX_STUDENTS = 500
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_SUBJECTS = 40
MAX_REF = 100


class RequestError(ValueError):
    """The request as a whole can't be answered (exit code 2)."""


@dataclass
class _Item:
    ref: str
    key: str
    tests_used: int


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_REF:
        raise RequestError(f"{what} must be a short text (1-{MAX_REF} characters).")
    return value


def _percent(value: Any, what: str) -> float | None:
    """A percentage 0-100, or None. Raises ValueError (refuses this subject only) when it is not one."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number from 0 to 100, or null.")
    if not 0 <= value <= 100 or not math.isfinite(value):
        raise ValueError(f"{what} must be from 0 to 100.")
    return float(value)


def _reason(r) -> dict:
    return {"key": r.key, "words": r.words, "value": r.value_text, "effect": r.effect, "sentence": r.sentence}


def _result(item: _Item, res: RiskResult) -> dict:
    return {"ref": item.ref, "key": item.key, "subject": res.subject, "percent": res.percent, "text": res.text,
            "reasons": [_reason(r) for r in res.reasons], "assumed": res.assumed, "known_subject": res.known_subject,
            "tests_used": item.tests_used}


def model_info(loaded: LoadedModel) -> dict:
    card = loaded.card
    return {
        "version": card.get("versions", {}).get("stud_ai"),
        "created": card.get("created"),
        "trees": card.get("trees"),
        "subjects_trained_on": card.get("subjects_trained_on", []),
        "file_sha256": loaded.model_sha256,
        "card_sha256": loaded.card_sha256,
    }


def answer(request: Any, loaded: LoadedModel | None = None) -> dict:
    """Work out the answer to one request (raises RequestError if the request as a whole is wrong)."""
    if not isinstance(request, dict):
        raise RequestError("The request must be a JSON object.")
    if type(request.get("contract")) is not int or request.get("contract") != CONTRACT:
        raise RequestError(f"Unknown contract {request.get('contract')!r}: this Stud-AI answers contract {CONTRACT}.")
    students = request.get("students")
    if not isinstance(students, list):
        raise RequestError('"students" must be a list.')
    if len(students) > MAX_STUDENTS:
        raise RequestError(f"Too many students in one request ({len(students)}; at most {MAX_STUDENTS}).")

    records, items, skipped, errors = [], [], [], []
    seen_refs: set[str] = set()
    for s, student in enumerate(students):
        if not isinstance(student, dict):
            raise RequestError(f"students[{s}] must be an object.")
        ref = _text(student.get("ref"), f"students[{s}].ref")
        if ref in seen_refs:
            raise RequestError(f"The student ref {ref!r} appears twice.")
        seen_refs.add(ref)
        subjects = student.get("subjects")
        if not isinstance(subjects, list) or len(subjects) > MAX_SUBJECTS:
            raise RequestError(f"students[{s}].subjects must be a list of at most {MAX_SUBJECTS} subjects.")

        marks: dict[str, SubjectMarks] = {}  # by subject name, in the order given
        keys: dict[str, str] = {}
        seen_keys: set[str] = set()
        for j, subject in enumerate(subjects):
            if not isinstance(subject, dict):
                raise RequestError(f"students[{s}].subjects[{j}] must be an object.")
            key = _text(subject.get("key"), f"students[{s}].subjects[{j}].key")
            if key in seen_keys:
                raise RequestError(f"The subject key {key!r} appears twice for student {ref!r}.")
            seen_keys.add(key)
            try:
                name = normalise_subject(_text(subject.get("name"), "The subject name"))
                tests = subject.get("tests")
                if not isinstance(tests, list):
                    raise ValueError("tests must be a list of percentages (it can be empty).")
                if len(tests) > MAX_TESTS:
                    raise ValueError(f"Send at most {MAX_TESTS} test scores (the latest ones).")
                scores = [_percent(t, "Each test score") for t in tests]
                if any(t is None for t in scores):
                    raise ValueError("Test scores can't be null: leave a missed test out of the list.")
                attendance = _percent(subject.get("attendance"), "attendance")
                homework = _percent(subject.get("homework"), "homework")
                if name in marks:
                    raise ValueError(f"Another subject of this student is also called {name!r}.")
            except ValueError as err:
                errors.append({"ref": ref, "key": key, "error": str(err)})
                continue
            marks[name] = SubjectMarks(tests=scores[-MAX_TESTS:], attendance=attendance, homework=homework)
            keys[name] = key

        if not marks:
            continue
        # One record per subject, each with the student's OTHER subjects worked out from the rest.
        for record in records_for_student(marks):
            item = _Item(ref=ref, key=keys[record.subject], tests_used=len(record.tests))
            if not record.tests:
                skipped.append({"ref": ref, "key": item.key, "reason": "no test scores yet"})
                continue
            records.append(record)
            items.append(item)

    # The model is loaded (and its pins checked) even when nothing needs predicting, so every
    # answer names the verified model: the portal refuses an answer without its hashes.
    loaded = loaded or load_model()
    if not records:
        return dict(contract=CONTRACT, model=model_info(loaded), results=[], skipped=skipped, errors=errors)
    try:
        results = predict_records(records, top_reasons=TOP_REASONS, loaded=loaded)
    except ValueError:
        # Something slipped past the checks above: find it, refuse just that subject, answer the rest.
        results, kept = [], []
        for item, record in zip(items, records):
            try:
                results.append(predict_records([record], top_reasons=TOP_REASONS, loaded=loaded)[0])
                kept.append(item)
            except ValueError as err:
                errors.append({"ref": item.ref, "key": item.key, "error": str(err)})
        items = kept
    return {"contract": CONTRACT, "model": model_info(loaded), "results": [_result(i, r) for i, r in zip(items, results)],
            "skipped": skipped, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stud-AI for the Campus Portal: JSON request in, risks out.")
    ap.add_argument("--in", dest="inp", type=Path, help="read the request from this file (default: standard input)")
    ap.add_argument("--out", type=Path, help="write the answer to this file (default: standard output)")
    args = ap.parse_args(argv)
    try:
        if args.inp:
            with args.inp.open("rb") as stream:
                raw_bytes = stream.read(MAX_REQUEST_BYTES + 1)
        else:
            raw_bytes = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw_bytes) > MAX_REQUEST_BYTES:
            raise RequestError("The request exceeds the 2 MiB limit.")
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise RequestError("The request must be UTF-8.") from None
        try:
            request = json.loads(raw)
        except (ValueError, RecursionError):
            raise RequestError("The request is not valid JSON.") from None
        response = answer(request)
    except RequestError as err:
        print(f"stud-ai: {err}", file=sys.stderr)
        return 2
    except (IntegrityError, ValueError) as err:
        print(f"stud-ai: {err}", file=sys.stderr)
        return 4
    except FileNotFoundError as err:
        print(f"stud-ai: {err}", file=sys.stderr)
        return 3
    text = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.buffer.write(text.encode("utf-8"))  # always UTF-8, whatever the server's language settings
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
