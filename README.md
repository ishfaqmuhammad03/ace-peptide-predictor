# ACE-Inhibitory Peptide Predictor — web server

A small, self-contained web server that predicts whether a peptide inhibits
angiotensin-converting enzyme (ACE), using the final trained model from our
study. Paste sequences into the page and it returns, for each one, a
probability of activity, a class call, and an applicability-domain check.

This repository contains **only the prediction server**: the trained model and
the code needed to run it. It is provided so that readers and reviewers can
reproduce the predictions reported in the paper.

---

## What the model does

| | |
|---|---|
| Model | Extremely randomised trees (`ExtraTreesClassifier`) |
| Input | Peptide sequence, 2–50 residues, standard amino acids |
| Descriptors | 879 sequence-derived descriptors, computed from the sequence alone |
| Positive class | ACE-inhibitory, IC50 < 50 µM |
| Decision threshold | 0.5 |
| Applicability domain | Mean Euclidean distance (standardised feature space) to the 5 nearest training peptides; inside if ≤ 28.606 |

Held-out test performance and the exact hyperparameters are recorded in
[`model/model_info.json`](model/model_info.json).

A prediction for a peptide **outside** the applicability domain is flagged on
the results page. Such a peptide is unlike anything the model was trained on,
so its probability should not be trusted at face value.

---

## Quick start

Requires Python 3.12 or 3.13.

```bash
git clone https://github.com/ishfaqmuhammad03/ace-peptide-predictor.git
cd ace-peptide-predictor

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --port 8000
```

Then open <http://localhost:8000> in a browser.

With conda instead:

```bash
conda env create -f environment.yml
conda activate webserver
uvicorn app:app --port 8000
```

With Docker:

```bash
docker build -t ace-peptide-predictor .
docker run -p 7860:7860 ace-peptide-predictor
```

> **`scikit-learn` must be version 1.9.1.** The model file was written with that
> version; loading it under a different one may fail or change the numbers.
> `requirements.txt` and `environment.yml` both pin it.

---

## Checking that your copy gives the right answers

```bash
python selftest.py
```

This runs eight reference peptides through the full pipeline and compares the
output against the probabilities recorded in this repository. All eight must
report `OK`. If any report `MISMATCH`, your environment differs from the one
the model was built in — check the `scikit-learn` version first.

---

## Using the page

Paste sequences into the box, one per line. FASTA and `name,SEQUENCE` are also
accepted, and a plain-text file can be uploaded. Up to 500 sequences per
request. Click **Load example** for a ready-made set, or **Download example
file** to see the accepted file format.

Results can be sorted by clicking a column heading and exported with
**Download CSV**.

## Programmatic access

```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"sequences": ["VVYPW", "IA", "LFP"]}'
```

Each result object contains `sequence`, `probability`, `label`, `in_domain`,
`ad_distance` and `note`. Interactive API documentation is served at
<http://localhost:8000/docs>.

---

## Files

```
app.py                     FastAPI application and prediction endpoint
features.py                Assembles the 879-descriptor vector for a sequence
pepfeat.py                 Descriptor definitions
pepfeat2.py                Descriptor definitions
peptide_features_103.py    Descriptor definitions
selftest.py                Reference-value check
model/                     Trained model, applicability-domain reference,
                           feature list and model metadata
static/                    Web page (HTML, CSS, JavaScript)
requirements.txt           pip dependencies
environment.yml            conda environment
Dockerfile                 Container build
```

---

## Citation

If you use this server, please cite the paper. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

> AUTHORS. TITLE. JOURNAL, YEAR. DOI

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
