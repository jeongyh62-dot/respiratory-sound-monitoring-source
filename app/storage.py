from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any


class SQLiteMeasurementStore:
    """SQLite repository for longitudinal respiratory-sound measurements.

    Raw WAV and XAI matrices are intentionally not embedded in the database.
    The database stores subject/file provenance and the monitoring trend label
    so that exported/merged datasets remain traceable.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    measurement_id TEXT NOT NULL UNIQUE,
                    subject_id TEXT NOT NULL,
                    measurement_time TEXT NOT NULL,
                    source_filename TEXT,
                    predicted_class TEXT,
                    confidence REAL NOT NULL,
                    p_normal REAL NOT NULL,
                    p_crackle REAL NOT NULL,
                    p_wheeze REAL NOT NULL,
                    p_both REAL NOT NULL,
                    abnormal_probability REAL NOT NULL,
                    trend_direction TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Safe in-place migration from the v12 database schema.
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(measurements)").fetchall()
            }
            if "trend_direction" not in columns:
                conn.execute("ALTER TABLE measurements ADD COLUMN trend_direction TEXT")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_measurements_subject_time "
                "ON measurements(subject_id, measurement_time, id)"
            )
            conn.execute("PRAGMA user_version = 2")

    def save_measurement(
        self,
        *,
        subject_id: str,
        measurement_id: str,
        measurement_time: str,
        source_filename: str | None,
        predicted_class: str | None,
        confidence: float,
        probabilities: dict[str, float],
        abnormal_probability: float,
        trend_direction: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO measurements (
                    measurement_id, subject_id, measurement_time, source_filename,
                    predicted_class, confidence,
                    p_normal, p_crackle, p_wheeze, p_both,
                    abnormal_probability, trend_direction, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    measurement_id,
                    subject_id,
                    measurement_time,
                    source_filename,
                    predicted_class,
                    float(confidence),
                    float(probabilities["Normal"]),
                    float(probabilities["Crackle"]),
                    float(probabilities["Wheeze"]),
                    float(probabilities["Both"]),
                    float(abnormal_probability),
                    trend_direction,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def update_trend_direction(self, measurement_id: str, trend_direction: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE measurements SET trend_direction = ? WHERE measurement_id = ?",
                (trend_direction, measurement_id),
            )

    def load_measurements(self, subject_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT measurement_id, subject_id, measurement_time, predicted_class, confidence,
                       p_normal, p_crackle, p_wheeze, p_both, abnormal_probability,
                       source_filename, trend_direction
                FROM measurements
                WHERE subject_id = ?
                ORDER BY id ASC
                """,
                (subject_id,),
            ).fetchall()

        return [
            {
                "measurement_id": row["measurement_id"],
                "subject_id": row["subject_id"],
                "measurement_time": row["measurement_time"],
                "source_filename": row["source_filename"],
                "Normal": float(row["p_normal"]),
                "Crackle": float(row["p_crackle"]),
                "Wheeze": float(row["p_wheeze"]),
                "Both": float(row["p_both"]),
                "Abnormal_Probability": float(row["abnormal_probability"]),
                "Predicted_Class": row["predicted_class"],
                "Confidence": float(row["confidence"]),
                "Trend_Direction": row["trend_direction"],
            }
            for row in rows
        ]

    def count_measurements(self, subject_id: str | None = None) -> int:
        with self._connect() as conn:
            if subject_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM measurements").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM measurements WHERE subject_id = ?",
                    (subject_id,),
                ).fetchone()
        return int(row["n"])

    def list_subjects(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT subject_id FROM measurements ORDER BY subject_id"
            ).fetchall()
        return [str(row["subject_id"]) for row in rows]
    def list_subject_summaries(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT subject_id, COUNT(*) AS measurement_count,
                       MAX(measurement_time) AS latest_measurement_time
                FROM measurements
                GROUP BY subject_id
                ORDER BY subject_id
                """
            ).fetchall()
        return [
            {
                "subject_id": str(row["subject_id"]),
                "measurement_count": int(row["measurement_count"]),
                "latest_measurement_time": row["latest_measurement_time"],
            }
            for row in rows
        ]

    def get_measurement(self, subject_id: str, measurement_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT measurement_id, subject_id, measurement_time, source_filename,
                       predicted_class, confidence, p_normal, p_crackle, p_wheeze, p_both,
                       abnormal_probability, trend_direction
                FROM measurements
                WHERE subject_id = ? AND measurement_id = ?
                """,
                (subject_id, measurement_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "measurement_id": row["measurement_id"],
            "subject_id": row["subject_id"],
            "measurement_time": row["measurement_time"],
            "source_filename": row["source_filename"],
            "predicted_class": row["predicted_class"],
            "confidence": float(row["confidence"]),
            "Normal": float(row["p_normal"]),
            "Crackle": float(row["p_crackle"]),
            "Wheeze": float(row["p_wheeze"]),
            "Both": float(row["p_both"]),
            "Abnormal_Probability": float(row["abnormal_probability"]),
            "Trend_Direction": row["trend_direction"],
        }

    def delete_measurement(self, subject_id: str, measurement_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM measurements WHERE subject_id = ? AND measurement_id = ?",
                (subject_id, measurement_id),
            )
        return cur.rowcount > 0

    def delete_subject(self, subject_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM measurements WHERE subject_id = ?",
                (subject_id,),
            )
        return int(cur.rowcount)

