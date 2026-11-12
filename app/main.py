from __future__ import annotations

from datetime import datetime, timedelta
import json
import re
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import (
    BASELINE_N,
    ROLLING_WINDOW,
    SAMPLE_RATE,
    TARGET_DURATION,
    CLASS_NAMES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    DATABASE_PATH,
    EXPORT_DIR,
)
from .monitoring import PatientMonitoringEngine
from .storage import SQLiteMeasurementStore

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(
    title="개인 맞춤형 호흡음 모니터링",
    version="0.2.0",
    description="Research PoC for respiratory sound classification and longitudinal monitoring.",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Runtime cache is hydrated from SQLite on first access for each subject.
# SQLite is the source of truth from v12 onward.
engines: dict[str, PatientMonitoringEngine] = {}
_model_service = None
_gradcam = None
_store = SQLiteMeasurementStore(DATABASE_PATH)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_model_service():
    global _model_service
    if _model_service is None:
        from .model_service import AlexNetRespiratoryClassifier
        _model_service = AlexNetRespiratoryClassifier()
    return _model_service


def get_gradcam():
    global _gradcam
    if _gradcam is None:
        from .gradcam_service import GradCAM
        model_service = get_model_service()
        _gradcam = GradCAM(model_service.model, model_service.model.features[10])
    return _gradcam


def build_xai_payload(model_service, result: dict) -> dict:
    import torch.nn.functional as F

    mel = result["mel"].detach().cpu()
    target_idx = CLASS_NAMES.index(result["predicted_class"])
    x = model_service.prepare_pretrained_input(mel).to(model_service.device)
    cam, _ = get_gradcam().generate(x, target_class=target_idx)

    cam_native = F.interpolate(
        cam[None, None, ...],
        size=mel.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu()

    mel_2d = [[round(float(v), 4) for v in row] for row in mel[0].tolist()]
    cam_2d = [[round(float(v), 4) for v in row] for row in cam_native.tolist()]

    return {
        "mel": mel_2d,
        "cam": cam_2d,
        "target_class": result["predicted_class"],
        "target_probability": float(result["confidence"]),
        "target_layer": "AlexNet features[10]",
        "sample_rate": SAMPLE_RATE,
        "duration_sec": TARGET_DURATION,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "fmin": FMIN,
        "fmax": FMAX,
    }


def _hydrate_engine(subject_id: str) -> PatientMonitoringEngine:
    engine = PatientMonitoringEngine(BASELINE_N, ROLLING_WINDOW)
    for row in _store.load_measurements(subject_id):
        engine.add_measurement(
            measurement_id=row["measurement_id"],
            probs={
                "Normal": row["Normal"],
                "Crackle": row["Crackle"],
                "Wheeze": row["Wheeze"],
                "Both": row["Both"],
            },
            measurement_time=row["measurement_time"],
            predicted_class=row["Predicted_Class"],
            confidence=row["Confidence"],
            subject_id=row.get("subject_id") or subject_id,
            source_filename=row.get("source_filename"),
        )

    # v13: backfill/migrate per-measurement trend labels for existing v12 rows.
    df = engine.get_dataframe()
    if not df.empty and "Trend_Direction" in df.columns:
        for _, r in df.iterrows():
            _store.update_trend_direction(
                str(r["measurement_id"]),
                None if r["Trend_Direction"] != r["Trend_Direction"] else str(r["Trend_Direction"]),
            )
    return engine


def get_engine(subject_id: str) -> PatientMonitoringEngine:
    if subject_id not in engines:
        engines[subject_id] = _hydrate_engine(subject_id)
    return engines[subject_id]


def _safe_subject_name(subject_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", subject_id.strip())
    return cleaned or "subject"


EXPORT_COLUMNS = [
    "measurement_id",
    "subject_id",
    "measurement_time",
    "source_filename",
    "Normal",
    "Crackle",
    "Wheeze",
    "Both",
    "Abnormal_Probability",
    "Predicted_Class",
    "Confidence",
    "Personal_Baseline",
    "Deviation_From_Baseline",
    "RollingMean",
    "Trend_Direction",
]


def _export_dataframe(subject_id: str, engine: PatientMonitoringEngine):
    df = engine.get_dataframe().copy()
    if df.empty:
        return df
    if "subject_id" not in df.columns:
        df["subject_id"] = subject_id
    else:
        df["subject_id"] = df["subject_id"].fillna(subject_id)
    if "source_filename" not in df.columns:
        df["source_filename"] = None
    return df.reindex(columns=EXPORT_COLUMNS)


def _write_csv_snapshot(subject_id: str, engine: PatientMonitoringEngine) -> Path | None:
    df = _export_dataframe(subject_id, engine)
    if df.empty:
        return None
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORT_DIR / f"{_safe_subject_name(subject_id)}_measurements.csv"
    temp = target.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(target)
    return target


def empty_summary():
    return {
        "measurement_id": "-",
        "timestamp": "-",
        "predicted_class": "-",
        "confidence": 0.0,
        "abnormal_probability": 0.0,
        "baseline": None,
        "deviation": None,
        "rolling_mean": None,
        "recent_slope": None,
        "recent_slope_unit": None,
        "trend_direction": "Pending",
        "trend_direction_ko": "산출 대기",
        "baseline_ready": False,
        "measurement_count": 0,
        "measurements_to_baseline": BASELINE_N,
    }


def build_context(subject_id: str, seed_demo: bool = False):
    engine = get_engine(subject_id)
    if seed_demo and not engine.records:
        now = datetime.now()
        demo = [
            ("M01", now - timedelta(days=8), {"Normal": 0.82, "Crackle": 0.08, "Wheeze": 0.07, "Both": 0.03}, "Normal", 0.82),
            ("M02", now - timedelta(days=6), {"Normal": 0.78, "Crackle": 0.10, "Wheeze": 0.08, "Both": 0.04}, "Normal", 0.78),
            ("M03", now - timedelta(days=4), {"Normal": 0.70, "Crackle": 0.14, "Wheeze": 0.11, "Both": 0.05}, "Normal", 0.70),
            ("M04", now - timedelta(days=2), {"Normal": 0.58, "Crackle": 0.20, "Wheeze": 0.15, "Both": 0.07}, "Normal", 0.58),
            ("M05", now, {"Normal": 0.49, "Crackle": 0.24, "Wheeze": 0.18, "Both": 0.09}, "Normal", 0.49),
        ]
        # Demo records remain intentionally ephemeral and are never written to SQLite.
        for mid, ts, probs, pred, conf in demo:
            engine.add_measurement(
                mid, probs, ts.isoformat(timespec="minutes"), pred, conf,
                subject_id=subject_id, source_filename="DEMO"
            )

    df = engine.get_dataframe()
    summary = engine.get_summary() or empty_summary()
    history = []
    if not df.empty:
        for _, r in df.iloc[::-1].iterrows():
            history.append({
                "measurement_id": r["measurement_id"],
                "timestamp": str(r["measurement_time"]),
                "source_filename": r.get("source_filename") or "-",
                "trend_direction": r.get("Trend_Direction") or "Pending",
                "predicted_class": r.get("Predicted_Class") or "-",
                "confidence": float(r.get("Confidence") or 0.0),
                "normal": float(r["Normal"]),
                "crackle": float(r["Crackle"]),
                "wheeze": float(r["Wheeze"]),
                "both": float(r["Both"]),
                "abnormal_probability": float(r["Abnormal_Probability"]),
                "baseline": None if r["Personal_Baseline"] != r["Personal_Baseline"] else float(r["Personal_Baseline"]),
                "rolling_mean": None if r["RollingMean"] != r["RollingMean"] else float(r["RollingMean"]),
                "deviation": None if r["Deviation_From_Baseline"] != r["Deviation_From_Baseline"] else float(r["Deviation_From_Baseline"]),
            })
    return {
        "subject_id": subject_id,
        "latest": summary,
        "history": history,
        "subjects": _store.list_subject_summaries(),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, subject_id: str = "SUBJ-TEST01", demo: int = 0):
    context = build_context(subject_id, seed_demo=bool(demo))
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context,
    )



def _infer_true_label_from_filename(filename: str) -> str | None:
    """Infer the project 4-class label from processed-cycle filenames.

    Expected examples include ``*_cycle00_Crackle.wav``. This helper is used
    only by the research batch-evaluation endpoint; it never changes or writes
    longitudinal subject records.
    """
    stem = Path(filename).stem.lower()
    for label in CLASS_NAMES:
        if stem.endswith(f"_{label.lower()}"):
            return label
    return None


def _batch_metrics(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("true_label") in CLASS_NAMES]
    if not labeled:
        return {
            "labeled_count": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": {},
        }

    confusion = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
    for row in labeled:
        confusion[row["true_label"]][row["predicted_class"]] += 1

    correct = sum(confusion[c][c] for c in CLASS_NAMES)
    per_class = {}
    recalls = []
    f1s = []
    for c in CLASS_NAMES:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        fp = sum(confusion[t][c] for t in CLASS_NAMES if t != c)
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / support if support else None
        f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
        per_class[c] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if recall is not None:
            recalls.append(recall)
        if f1 is not None:
            f1s.append(f1)

    return {
        "labeled_count": len(labeled),
        "accuracy": correct / len(labeled),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


@app.post("/api/batch-evaluate")
def batch_evaluate(
    files: list[UploadFile] = File(...),
    infer_labels_from_filename: bool = Form(True),
):
    """Evaluate multiple respiratory-cycle WAVs without modifying longitudinal storage.

    This endpoint is intentionally separated from ``/api/analyze-cycle``: batch
    evaluation is a research/evaluation workflow and must not contaminate a
    subject's SQLite baseline or monitoring history.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one WAV file is required.")
    if len(files) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 WAV files per batch request.")

    started = time.perf_counter()
    model = get_model_service()
    rows = []
    errors = []

    for idx, file in enumerate(files, start=1):
        filename = file.filename or f"batch_{idx:03d}.wav"
        if not filename.lower().endswith(".wav"):
            errors.append({"filename": filename, "error": "Only WAV files are accepted."})
            continue
        try:
            file.file.seek(0)
            result = model.predict_upload(file.file, filename)
            true_label = _infer_true_label_from_filename(filename) if infer_labels_from_filename else None
            row = {
                "filename": filename,
                "true_label": true_label,
                "predicted_class": result["predicted_class"],
                "confidence": float(result["confidence"]),
                "Normal": float(result["probabilities"]["Normal"]),
                "Crackle": float(result["probabilities"]["Crackle"]),
                "Wheeze": float(result["probabilities"]["Wheeze"]),
                "Both": float(result["probabilities"]["Both"]),
                "Abnormal_Probability": float(result["abnormal_probability"]),
            }
            row["correct"] = None if true_label is None else (true_label == row["predicted_class"])
            rows.append(row)
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})

    metrics = _batch_metrics(rows)
    return {
        "file_count": len(files),
        "success_count": len(rows),
        "error_count": len(errors),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "labels_inferred_from_filename": bool(infer_labels_from_filename),
        "class_order": CLASS_NAMES,
        "metrics": metrics,
        "results": rows,
        "errors": errors,
        "notice": "Research batch evaluation only. Results are not written to longitudinal SQLite/CSV subject history.",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "AlexNet Base",
        "sample_rate": SAMPLE_RATE,
        "target_duration_sec": TARGET_DURATION,
        "baseline_n": BASELINE_N,
        "rolling_window": ROLLING_WINDOW,
        "persistence": "SQLite + CSV snapshot",
        "stored_measurements": _store.count_measurements(),
    }


@app.get("/api/storage-status")
def storage_status():
    return {
        "database": str(DATABASE_PATH),
        "database_exists": DATABASE_PATH.exists(),
        "export_directory": str(EXPORT_DIR),
        "measurement_count": _store.count_measurements(),
        "subjects": _store.list_subjects(),
    }


@app.get("/api/subjects/{subject_id}/measurements")
def subject_measurements(subject_id: str):
    context = build_context(subject_id)
    return {
        "subject_id": subject_id,
        "measurement_count": len(context["history"]),
        "latest": context["latest"],
        "measurements": list(reversed(context["history"])),
    }


@app.get("/api/export/{subject_id}.csv")
def export_subject_csv(subject_id: str):
    engine = get_engine(subject_id)
    df = _export_dataframe(subject_id, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="No measurements found for this subject.")
    _write_csv_snapshot(subject_id, engine)
    csv_text = df.to_csv(index=False)
    filename = f"{_safe_subject_name(subject_id)}_measurements.csv"
    return Response(
        content="\ufeff" + csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/export/{subject_id}.json")
def export_subject_json(subject_id: str):
    engine = get_engine(subject_id)
    df = _export_dataframe(subject_id, engine)
    if df.empty:
        raise HTTPException(status_code=404, detail="No measurements found for this subject.")
    records = df.where(df.notna(), None).to_dict(orient="records")
    filename = f"{_safe_subject_name(subject_id)}_measurements.json"
    return Response(
        content=json.dumps(records, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _refresh_subject_after_mutation(subject_id: str) -> dict:
    engines.pop(subject_id, None)
    engine = get_engine(subject_id)

    # Recalculate and synchronize longitudinal trend labels after deletion.
    try:
        df = engine.get_dataframe()
        if not df.empty and "Trend_Direction" in df.columns:
            for _, row in df.iterrows():
                _store.update_trend_direction(
                    str(row["measurement_id"]),
                    str(row["Trend_Direction"]),
                )
    except Exception as exc:
        print(f"[storage] trend refresh FAILED: {exc}", flush=True)

    snapshot = EXPORT_DIR / f"{_safe_subject_name(subject_id)}_measurements.csv"
    if engine.records:
        try:
            _write_csv_snapshot(subject_id, engine)
        except Exception as exc:
            print(f"[storage] CSV refresh FAILED: {exc}", flush=True)
    else:
        snapshot.unlink(missing_ok=True)

    context = build_context(subject_id)
    return {
        "subject_id": subject_id,
        "measurement_count": len(context["history"]),
        "latest": context["latest"],
        "measurements": list(reversed(context["history"])),
    }


@app.get("/api/subjects")
def list_subjects():
    return {"subjects": _store.list_subject_summaries()}


@app.get("/api/subjects/{subject_id}/measurements/{measurement_id}")
def get_subject_measurement(subject_id: str, measurement_id: str):
    row = _store.get_measurement(subject_id, measurement_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Measurement not found.")
    return row


@app.delete("/api/subjects/{subject_id}/measurements/{measurement_id}")
def delete_subject_measurement(subject_id: str, measurement_id: str):
    if not _store.delete_measurement(subject_id, measurement_id):
        raise HTTPException(status_code=404, detail="Measurement not found.")
    payload = _refresh_subject_after_mutation(subject_id)
    payload["deleted_measurement_id"] = measurement_id
    return payload


@app.delete("/api/subjects/{subject_id}")
def delete_subject_records(subject_id: str):
    deleted_count = _store.delete_subject(subject_id)
    engines.pop(subject_id, None)
    snapshot = EXPORT_DIR / f"{_safe_subject_name(subject_id)}_measurements.csv"
    snapshot.unlink(missing_ok=True)
    return {
        "subject_id": subject_id,
        "deleted_count": deleted_count,
        "measurement_count": 0,
    }


@app.post("/api/analyze-cycle")
def analyze_cycle(
    file: UploadFile = File(...),
    subject_id: str = Form("SUBJ-DEMO"),
    measurement_time: str | None = Form(None),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are accepted.")
    request_started = time.perf_counter()
    print(f"[analyze] START subject={subject_id} file={file.filename}", flush=True)
    try:
        model = get_model_service()
        result = model.predict_upload(file.file, file.filename)
        print(f"[analyze] inference OK {time.perf_counter()-request_started:.3f}s", flush=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    try:
        xai = build_xai_payload(model, result)
        print(f"[analyze] xai OK {time.perf_counter()-request_started:.3f}s", flush=True)
    except Exception as e:
        print(f"[analyze] xai FAILED: {e}", flush=True)
        xai = None

    engine = get_engine(subject_id)
    measurement_id = f"M{len(engine.records)+1:03d}-{uuid4().hex[:6]}"
    ts = measurement_time or datetime.now().isoformat(timespec="seconds")
    engine.add_measurement(
        measurement_id=measurement_id,
        probs=result["probabilities"],
        measurement_time=ts,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        subject_id=subject_id,
        source_filename=file.filename,
    )
    summary = engine.get_summary()

    try:
        _store.save_measurement(
            subject_id=subject_id,
            measurement_id=measurement_id,
            measurement_time=ts,
            source_filename=file.filename,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            abnormal_probability=result["abnormal_probability"],
            trend_direction=summary["trend_direction"] if summary else None,
        )
    except Exception as e:
        # SQLite is the source of truth. Roll back the runtime append only when
        # the database insert itself fails.
        if engine.records and engine.records[-1].get("measurement_id") == measurement_id:
            engine.records.pop()
        raise HTTPException(status_code=500, detail=f"SQLite persistence failed: {e}") from e

    csv_path = None
    try:
        csv_path = _write_csv_snapshot(subject_id, engine)
    except Exception as e:
        # A CSV snapshot is a secondary convenience copy. Do not discard a
        # successfully committed SQLite record if snapshot refresh fails.
        print(f"[analyze] CSV snapshot FAILED: {e}", flush=True)

    # Keep trend labels in SQLite synchronized for both new and migrated rows.
    try:
        df_sync = engine.get_dataframe()
        if not df_sync.empty and "Trend_Direction" in df_sync.columns:
            for _, r in df_sync.iterrows():
                _store.update_trend_direction(str(r["measurement_id"]), str(r["Trend_Direction"]))
    except Exception as e:
        print(f"[analyze] trend sync FAILED: {e}", flush=True)

    print(
        f"[analyze] monitoring OK count={len(engine.records)} "
        f"baseline_ready={summary['baseline_ready']} "
        f"persisted=sqlite+csv total={time.perf_counter()-request_started:.3f}s",
        flush=True,
    )
    return {
        "measurement_id": measurement_id,
        "subject_id": subject_id,
        "prediction": {k: v for k, v in result.items() if k != "mel"},
        "monitoring": summary,
        "xai": xai,
        "persistence": {
            "sqlite": True,
            "csv_snapshot": str(csv_path) if csv_path else None,
        },
        "notice": "Research prototype only. This endpoint expects one respiratory-cycle WAV; full-recording cycle segmentation is not included yet.",
    }
