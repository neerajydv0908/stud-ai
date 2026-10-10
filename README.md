# Stud-AI

Hi there, and thanks for stopping by! 👋

I'm **Neeraj Yadav**, a master's student in Computer Science at the University of Kansas. From January 2022 to December 2024 I worked at **AESRI (Aryavart Educational and Social Research Institute)**, which runs two schools in Bikaner, Rajasthan. I built these applications for the schools and maintained them.

The project has three parts, each in its own repository:

| Part | What it does |
| --- | --- |
| **[Website](https://github.com/neerajydv0908/aesri-website)** | The public website for AESRI and its two schools |
| **[Campus Portal](https://github.com/neerajydv0908/aesri-portal)** | The private portal where families, teachers, the school offices and AESRI sign in |
| **[Stud-AI](https://github.com/neerajydv0908/stud-ai)** (you are here) | My machine-learning model that helps class teachers see which students may need support |

Let me walk you through Stud-AI.

---

## What it does

Stud-AI estimates, **for every subject**, how likely a student is not to pass the next examination in it, and says why. Here's one student, three subjects:

```text
Estimated risk of not passing the next mathematics examination: 63%
  - Mathematics test scores so far: average 41, latest 41 (out of 100, 3 tests), falling about 2 marks per test (raises the risk)
  - Average test score in other subjects: 56 out of 100 (lowers the risk)
  - Homework done in mathematics: 55% (raises the risk)
Estimated risk of not passing the next English examination: 4%
  - English test scores so far: average 62, latest 66 (out of 100, 3 tests), rising about 2 marks per test (lowers the risk)
  - Average test score in other subjects: 46 out of 100 (raises the risk)
  - Homework done in other subjects: 62% (raises the risk)
Estimated risk of not passing the next science examination: 39%
  - Science test scores so far: average 50, latest 52 (out of 100, 3 tests) (raises the risk)
  - Homework done in other subjects: 68% (raises the risk)
  - Homework done in science: 70% (raises the risk)
```

In the [Campus Portal](https://github.com/neerajydv0908/aesri-portal), class teachers see these estimates in a *Needs support* view, so they can spot early in the year which students may need help in which subject. Families and students never see them.

## How it works

- **One CatBoostClassifier for every subject.** The subject is one of its inputs, so a single model covers them all.
- **Trained on five subjects:** mathematics, English, science, history and geography, from the dataset I trained it on. It reads the subjects from the data's column names, so retraining on AESRI's own records with Hindi, Sanskrit, EVS or social science needs no code change.
- **Runs locally.** It runs on your own computer or server: no AI service, no API key, and no student data leaves the school. The Campus Portal runs it on its own server (see [For the Campus Portal](#for-the-campus-portal)).

---

## How good is it?

I scored it once on `test_students.csv`: 750 students the model never saw while training or choosing settings, in 5 subjects (3,750 subject results), with all 5 tests known.

| Measure | Test set | Validation set |
| --- | ---: | ---: |
| ROC-AUC (0.5 = guessing, 1 = perfect) | 0.933 | 0.927 |
| Accuracy at a 50% cut-off | 86.7% | 86.4% |
| Results that were not a pass, caught at 50% | 74.7% | 70.6% |
| Right when it says 50% or more | 81.2% | 78.4% |
| Brier score (lower is better) | 0.095 | 0.095 |

By subject (test set, ROC-AUC):

| Mathematics | English | Science | History | Geography |
| ---: | ---: | ---: | ---: | ---: |
| 0.942 | 0.941 | 0.938 | 0.923 | 0.910 |

- **With fewer tests known, it's less sure, as it should be.** ROC-AUC is 0.879 with 1 test, 0.894 with 2, 0.904 with 3 and 0.918 with 4 (test set).
- **A stated risk means what it says.** When it says "30%", about 30% of those results really were not a pass (calibration error 1.4%).
- **Girls and boys, and every age group, get the same quality.** The model is never told either.
- **What to expect in general:** across 5 cross-validation folds of the other 4,250 students, about 0.927 ROC-AUC and 86% accuracy with 5 tests known. The test set came out slightly on the lucky side.

I keep the full reports private (they aren't in this repository): the test report with tables and charts (`reports/test-report.md`), the settings I tried on the validation set (`reports/validation-search.md`), and the design experiments (`reports/improvement-experiments.md`).

**Why one model for all subjects?** I compared it with cross-validation against a separate model for each subject, a model that isn't told the subject, a model without the "doing better never raises the risk" rules, and logistic regression. Its probabilities were better than each of theirs (lower log loss) in all 5 folds. The subject matters: in this dataset, English is much easier to pass at the same marks than the other four subjects.

**Can it do better?** Not much with these columns. Around a subject average of 45–55, students who pass and students who don't look almost the same on homework and attendance. What would help is better inputs from the portal, and real records.

### Two honest notes

- **A simple logistic regression comes close on this dataset** (test ROC-AUC 0.934; in cross-validation its probabilities were a little worse in every fold). The data mostly follows straight-line patterns, so CatBoost has little extra to find here. On real school records it may help more, and it handles missing values and uneven data well.
- **It learned from the dataset I supplied, not the schools' own records.** Before teachers rely on it, it should be retrained and re-tested on AESRI's own past records (see [Training on AESRI's own records](#training-on-aesris-own-records)).

## What it looks at

| Used | Not used |
| --- | --- |
| The subject. In that subject: attendance %, homework done %, and the test scores so far (latest, average, lowest, highest, trend, how many). The other subjects' average test score, homework % and attendance % | Name, student ID, serial number, **gender, age**, and the other subjects' pass/fail (decided at the same time, so unknown before the exam) |

- **Works from the first test.** Each training student is shown with 1, 2, 3, 4 and 5 tests known, so the model can be used early in the year.
- **Doing better never raises the risk.** Higher marks, more homework or better attendance can only lower a student's risk, never raise it, all else equal. I checked this on all 25,000 validation rows, and the tests check it again every time they run.
- **Missing values** (for example, no homework recorded) are filled with a typical value from the training data. The output lists them, and they're never given as reasons.
- **Reasons** come from the model's own SHAP values (built into CatBoost): the inputs that moved this student's estimate the most. Tiny effects aren't shown.
- **A subject it wasn't trained on** (Hindi, say) gets no estimate, only "Not enough data as of now", without a model prediction. To estimate it, retrain and validate with that subject.

---

## Run it on your computer

You'll need **Python 3.10 or newer**. Check with `python3 --version`; if it's older, install Python from [python.org](https://www.python.org/downloads/).

```bash
cd Stud-AI
python3 -m venv .venv.nosync          # ".nosync" stops iCloud from copying it, if your folder is synced
source .venv.nosync/bin/activate
pip install -r requirements-dev.txt   # CatBoost, pandas, scikit-learn, matplotlib, pytest
```

The trained model is already in `models/`, so you can predict straight away:

```bash
python -m stud_ai.predict --subject science --tests 45 38 41 --attendance 72 --homework 55 --other-average 52
python -m stud_ai.predict --csv test_students.csv --tests-known 3 --out predictions.csv   # every student, every subject
```

From Python, every subject of one student at once (the "other subjects" inputs are worked out for you):

```python
from stud_ai.features import SubjectMarks
from stud_ai.predict import predict_student

results = predict_student({
    "mathematics": SubjectMarks(tests=[45, 38, 41], attendance=72, homework=55),
    "english": SubjectMarks(tests=[62, 58, 66], attendance=90, homework=80),
    "science": SubjectMarks(tests=[50, 47, 52], attendance=85, homework=70),
})
for result in results:
    print(result.text)      # Estimated risk of not passing the next mathematics examination: 63%
    print(result.percent)   # 63
    for reason in result.reasons:
        print("  -", reason.sentence)
```

Scores and percentages are out of 100, so convert marks first: 18 out of 25 is 72.

Train, test and check again:

```bash
python -m stud_ai.train --search    # tries settings on the validation set, then trains (about 3 minutes; leave out --search for 10 seconds)
# Review python -m stud_ai.artifact_hashes; supply both STUD_AI_*_SHA256 pins for the candidate
python -m stud_ai.evaluate          # writes reports and updates the card; review its new final hash
python -m pytest                    # 157 checks
```

Training is repeatable: the same data gives exactly the same model. Predictions are the same on any computer, and with the oldest and newest library versions allowed by `requirements.txt` (checked with CatBoost 1.2.7 and 1.2.10).

## Training on AESRI's own records

Export past students into three CSV files (training, validation and test), one row per student, and put them in a `data/` folder inside Stud-AI with the same names: `data/train_students.csv`, `data/validation_students.csv` and `data/test_students.csv`. Stud-AI reads `data/` first (when it holds any of the three files, all three come from there), and git never publishes it, so real student records stay off GitHub. Keep each student in only one file.

The columns:

- `student_id`
- for **every subject**, named in lower case with `_` between words (`mathematics`, `english`, `hindi`, `social_science`, `environmental_studies`, …):
  - `<subject>_attendance_percent`
  - `<subject>_homework_completion_percent`
  - `<subject>_test_1_score` … `<subject>_test_5_score` (out of 100, oldest first)
  - `<subject>_passed` (1 = passed, 0 = did not)
- optionally `gender` and `age`. The model never reads them; the test report uses them to check it works equally well for every group.

The subjects come from these column names, so no code change is needed, and the tests follow them too. All three files need the same subjects, and every cell must be filled in. Then run `train`, `evaluate` and the tests again, and read the new report before using it.

## For the Campus Portal

The portal runs Stud-AI on the same server, as a short program: `python -m stud_ai.batch` reads a request in JSON on standard input and writes the answer on standard output. It handles every student and subject at once, in about a second for a class. Nothing goes over the network, and Stud-AI is given only numbers under the portal's own IDs, never names.

```bash
echo '{"contract": 3, "students": [{"ref": "s1", "subjects": [
  {"key": "m", "name": "Mathematics", "tests": [45, 38, 41], "attendance": 72, "homework": 55},
  {"key": "e", "name": "English", "tests": [62, 58, 66], "attendance": 90, "homework": 80}]}]}' | python -m stud_ai.batch
```

- **In:** per student and subject, the test scores so far as percentages (oldest first; the latest 5 are used), attendance % and homework %, or `null` when not known.
- **Out:** per subject, the risk as a whole percent (`percent`), the sentence ("Estimated risk of not passing the next mathematics examination: 62%" for the example above), the main reasons in words, the inputs that were assumed, whether the model was trained on that subject, identifiers and `tests_used`. A subject it wasn't trained on gets `percent: null` and "Not enough data as of now for Hindi", without inference. Subjects without scores are skipped, and malformed subjects return structured errors. Requests are limited to 2 MiB, 500 students, 40 subjects each and five scores per subject.
- **Exit codes:** 0 answered, 2 invalid request, 3 missing artifact, 4 integrity/configuration failure. Errors are explained on standard error.

**Security.** The portal uses contract 3: for each subject it receives only the whole percent, its sentence and the main reasons in words, never raw probabilities, logits or SHAP weights. Both the model and its metadata are verified against trusted SHA-256 pins before they're loaded. The full contract, quotas and production deployment are in the portal's `docs/STUD-AI.md` (a private guide, not in the portal's repository).

In the portal, class teachers and admins see the estimates in *Needs support*: a student is red at an estimate of 50% or more in any subject, or under the separate attendance/concern rule. Families and students never see them.

## Promoting a new model

`python -m stud_ai.artifact_hashes` prints the candidate model and card SHA-256 hashes and never changes what's trusted. For development, review a newly trained pair yourself and update `stud_ai/trusted_artifacts.json`, or supply both `STUD_AI_MODEL_SHA256` and `STUD_AI_CARD_SHA256`. Evaluation updates the card, so repin the reviewed final pair before running the tests or deploying.

Production requires both hashes from the deployment configuration and never falls back to the development pins. Generate the hashes in the trusted build environment, and promote code, model, card and pins together. The deployed account must not be able to write them.

## Files

| Path | What it is |
| --- | --- |
| `train_students.csv`, `validation_students.csv`, `test_students.csv` | The dataset supplied with the project |
| `data/` | AESRI's own records, once you add them: read first, and never published (see above) |
| `stud_ai/features.py` | Turns a student's record into the model's inputs (the same code for training and prediction) |
| `stud_ai/train.py` | Trains on the training file, stops early on the validation file, saves the model |
| `stud_ai/evaluate.py` | Scores the test file once and writes `reports/` |
| `stud_ai/predict.py` | Risk and reasons for one subject, for every subject of one student, or for a whole CSV |
| `stud_ai/batch.py` | What the Campus Portal runs: many students at once, JSON in and out |
| `stud_ai/metrics.py`, `stud_ai/model.py` | Measures; model settings and loading |
| `models/pass_risk.cbm`, `models/model_card.json` | The trained model, and what it is: subjects, inputs, settings, data fingerprints, results |
| `reports/` | Test report, charts, every test student's prediction in every subject, settings and designs tried (only the charts are in this repository; the reports and predictions stay private) |
| `tests/` | Inputs, the sentence for every subject, "doing better never raises the risk", single vs batch, other subject names, the portal's JSON command, where the dataset files are read from |
| `experiments/cv.py` | The cross-validation I used to compare designs (`python experiments/cv.py`, about 4 minutes) |

Tested with Python 3.11, CatBoost 1.2.10, pandas 3.0.6, NumPy 2.4.6, scikit-learn 1.9.1 and matplotlib 3.11.2.

---

## Thanks for reading!

If you have questions or ideas about the project, you can find me on GitHub at [@neerajydv0908](https://github.com/neerajydv0908).

— Neeraj
