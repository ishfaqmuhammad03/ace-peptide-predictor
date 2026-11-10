import json
import re
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features import compute_feature_matrix
from peptide_features_103 import AA

BASE = Path(__file__).resolve().parent
INFO = json.loads((BASE / "model" / "model_info.json").read_text())

CONFIG = {
    "title": "ACE-Inhibitory Peptide Predictor",
    "version": "1.0",
    "class1_meaning": "Active",
    "class0_meaning": "Inactive",
    "activity_definition": "IC50 < 50 µM against angiotensin-converting enzyme",
    "threshold": 0.5,
    "min_length": 2,
    "max_length": 50,
    "max_sequences": 500,
    "model": INFO["model"],
    "model_display": "ExtraTrees classifier (extremely randomised trees)",
    "training": INFO["training"],
    "metrics": {"MCC": INFO["test_metrics"]["MCC"], "Accuracy": INFO["test_metrics"]["ACC"],
                "ROC-AUC": INFO["test_metrics"]["ROC_AUC"], "AUPRC": INFO["test_metrics"]["AUPRC"]},
    "citation": "Ishfaq, M. et al. Manuscript submitted. See CITATION.cff in the repository.",
    "contact": "Muhammad Ishfaq, Huanggang Normal University (muhammad@hgnu.edu.cn)",
}

FEATURES = json.loads((BASE / "model" / "feature_columns.json").read_text())
MODEL = joblib.load(BASE / "model" / "extratrees_model.joblib")
_AD = joblib.load(BASE / "model" / "ad_reference.joblib")
AD_SCALER, AD_NN, AD_THRESHOLD = _AD["scaler"], _AD["nn"], _AD["threshold"]

app = FastAPI(title=CONFIG["title"], version=CONFIG["version"])


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=300_000, description="FASTA, or one sequence per line")


def parse_input(text: str):
    """Return a list of (name, sequence). Accepts FASTA or plain lines ('SEQ' or 'name,SEQ')."""
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")
             if ln.strip() and not ln.lstrip().startswith("#")]
    items = []
    if any(ln.startswith(">") for ln in lines):
        name, parts = None, []
        for ln in lines:
            if ln.startswith(">"):
                if name is not None or parts:
                    items.append((name or f"seq{len(items) + 1}", "".join(parts)))
                name, parts = ln[1:].strip() or None, []
            else:
                parts.append(re.sub(r"\s+", "", ln))
        items.append((name or f"seq{len(items) + 1}", "".join(parts)))
    else:
        for ln in lines:
            tok = [t for t in re.split(r"[,\t;]+", ln) if t.strip()]
            seq = tok[-1].strip() if tok else ""
            name = tok[0].strip() if len(tok) > 1 else f"seq{len(items) + 1}"
            items.append((name, seq))
    return items


@app.get("/api/info")
def info():
    return {**CONFIG, "n_features": len(FEATURES), "ad_threshold": AD_THRESHOLD,
            "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__}}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    items = parse_input(req.text)
    if not items:
        return JSONResponse({"detail": "No sequences found. Paste one sequence per line or upload a FASTA file."}, 400)
    if len(items) > CONFIG["max_sequences"]:
        return JSONResponse({"detail": f"Too many sequences ({len(items)}). The limit is {CONFIG['max_sequences']} per run."}, 400)

    results, ok_idx = [], []
    for i, (name, raw) in enumerate(items):
        seq = raw.upper()
        row = {"id": i + 1, "name": name, "sequence": seq, "length": len(seq), "status": "ok",
               "probability": None, "label": None, "note": "", "in_domain": None, "ad_distance": None}
        bad = sorted(set(seq) - set(AA))
        if not seq:
            row.update(status="rejected", note="Empty sequence.")
        elif bad:
            row.update(status="rejected", note=f"Unsupported letters: {', '.join(bad)}. Use the 20 standard amino acids.")
        elif len(seq) < CONFIG["min_length"]:
            row.update(status="rejected", note=f"Too short. The minimum length is {CONFIG['min_length']}.")
        else:
            if len(seq) > CONFIG["max_length"]:
                row["note"] = f"Longer than the training range (max {CONFIG['max_length']}). Treat with caution."
            ok_idx.append(i)
        results.append(row)

    if ok_idx:
        try:
            X = compute_feature_matrix([results[i]["sequence"] for i in ok_idx], FEATURES)
        except Exception as exc:
            return JSONResponse({"detail": f"Feature calculation failed: {exc}"}, 500)
        Xv = X.to_numpy(dtype=float)
        if not np.isfinite(Xv).all():
            return JSONResponse({"detail": "Feature calculation produced invalid values."}, 500)
        prob = MODEL.predict_proba(Xv)[:, 1]
        dist = AD_NN.kneighbors(AD_SCALER.transform(Xv))[0][:, :5].mean(1)
        for k, i in enumerate(ok_idx):
            p = float(prob[k]); d = float(dist[k])
            results[i]["probability"] = round(p, 4)
            results[i]["label"] = int(p >= CONFIG["threshold"])
            results[i]["ad_distance"] = round(d, 3)
            results[i]["in_domain"] = bool(d <= AD_THRESHOLD)
            if not results[i]["in_domain"]:
                extra = "Outside the applicability domain - this peptide is unlike the training set, so the prediction is less reliable."
                results[i]["note"] = (results[i]["note"] + " " + extra).strip()

    n1 = sum(1 for r in results if r["label"] == 1)
    n0 = sum(1 for r in results if r["label"] == 0)
    return {"results": results,
            "summary": {"total": len(results), "class1": n1, "class0": n0,
                        "in_domain": sum(1 for r in results if r["in_domain"]),
                        "rejected": sum(1 for r in results if r["status"] == "rejected")}}


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
