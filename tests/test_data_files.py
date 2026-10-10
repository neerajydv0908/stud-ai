"""Where the dataset files are read from (stud_ai/model.py data_folder / data_file).

AESRI's own records go in data/, which git never publishes. When data/ holds any of the three files, all
three are read from there, so a model is never trained on one dataset and tested on another.
"""
from pathlib import Path

import pytest

from stud_ai import model
from stud_ai.model import DATA_FILES, data_file, data_folder

ROOT_FILES = list(DATA_FILES.values())


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A stand-in project folder with the supplied dataset in it."""
    for filename in ROOT_FILES:
        (tmp_path / filename).write_text("student_id\n", encoding="utf-8")
    monkeypatch.setattr(model, "ROOT", tmp_path)
    return tmp_path


def test_the_supplied_dataset_is_used_when_data_folder_is_empty_or_missing(project):
    assert data_folder() == project
    assert data_file("train") == project / "train_students.csv"
    (project / "data").mkdir()
    (project / "data" / "notes.txt").write_text("not a dataset file", encoding="utf-8")
    assert data_file("test") == project / "test_students.csv"


def test_own_records_in_data_are_read_first(project):
    own = project / "data"
    own.mkdir()
    for filename in ROOT_FILES:
        (own / filename).write_text("student_id\n", encoding="utf-8")
    assert data_folder() == own
    for name, filename in DATA_FILES.items():
        assert data_file(name) == own / filename


def test_a_partly_filled_data_folder_is_never_mixed_with_the_supplied_dataset(project):
    own = project / "data"
    own.mkdir()
    (own / "train_students.csv").write_text("student_id\n", encoding="utf-8")
    assert data_file("train") == own / "train_students.csv"
    with pytest.raises(FileNotFoundError, match="put train_students.csv, validation_students.csv, test_students.csv there together"):
        data_file("test")


def test_a_folder_asked_for_is_used_as_it_is(project, tmp_path_factory):
    other = tmp_path_factory.mktemp("elsewhere")
    (other / "validation_students.csv").write_text("student_id\n", encoding="utf-8")
    assert data_file("validation", other) == other / "validation_students.csv"
    with pytest.raises(FileNotFoundError, match=str(other)):
        data_file("train", other)


def test_git_never_publishes_own_records():
    """The data/ folder and files written from predictions stay out of git (see .gitignore)."""
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/data/" in ignore
    assert "*.csv" in ignore
    assert {f"!/{filename}" for filename in ROOT_FILES} <= set(ignore)
