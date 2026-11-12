from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PatientMonitoringEngine:
    baseline_n: int = 3
    rolling_window: int = 3
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_measurement(
        self,
        measurement_id: str,
        probs: dict[str, float],
        measurement_time: str | datetime | None = None,
        predicted_class: str | None = None,
        confidence: float | None = None,
        subject_id: str | None = None,
        source_filename: str | None = None,
    ) -> dict[str, Any]:
        abnormal_probability = float(probs["Crackle"] + probs["Wheeze"] + probs["Both"])
        record = {
            "measurement_id": str(measurement_id),
            "subject_id": subject_id,
            "measurement_time": measurement_time or datetime.now().isoformat(timespec="seconds"),
            "source_filename": source_filename,
            "Normal": float(probs["Normal"]),
            "Crackle": float(probs["Crackle"]),
            "Wheeze": float(probs["Wheeze"]),
            "Both": float(probs["Both"]),
            "Abnormal_Probability": abnormal_probability,
            "Predicted_Class": predicted_class,
            "Confidence": confidence,
        }
        self.records.append(record)
        return record

    def _slope_and_direction(self, recent: pd.DataFrame) -> tuple[float | None, str | None, str]:
        if len(recent) < 2:
            return None, None, "Pending"

        times = pd.to_datetime(recent["measurement_time"], errors="coerce")
        if times.notna().all() and times.nunique() >= 2:
            x = (times - times.iloc[0]).dt.total_seconds().to_numpy() / 86400.0
            slope_unit = "day"
        else:
            x = np.arange(len(recent), dtype=float)
            slope_unit = "measurement"

        y = recent["Abnormal_Probability"].to_numpy(dtype=float)
        x_range = float(x.max() - x.min())
        if x_range <= 0:
            return None, slope_unit, "Pending"

        x_mean = float(x.mean())
        y_mean = float(y.mean())
        numerator = float(((x - x_mean) * (y - y_mean)).sum())
        denominator = float(((x - x_mean) ** 2).sum())
        if denominator <= 0:
            return None, slope_unit, "Pending"

        slope = numerator / denominator
        if slope > 1e-12:
            direction = "Increasing"
        elif slope < -1e-12:
            direction = "Decreasing"
        else:
            direction = "Stable"
        return slope, slope_unit, direction

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.records)
        if df.empty:
            return df

        baseline_ready = len(df) >= self.baseline_n
        if baseline_ready:
            baseline_value = float(df.iloc[: self.baseline_n]["Abnormal_Probability"].mean())
            df["Personal_Baseline"] = baseline_value
            df["Deviation_From_Baseline"] = df["Abnormal_Probability"] - baseline_value
        else:
            df["Personal_Baseline"] = np.nan
            df["Deviation_From_Baseline"] = np.nan

        df["RollingMean"] = df["Abnormal_Probability"].rolling(
            window=self.rolling_window,
            min_periods=self.rolling_window,
        ).mean()

        # Preserve the monitoring interpretation associated with each measurement.
        # Before the baseline is formed the direction remains Pending. From the
        # baseline-forming measurement onward, the direction is calculated using
        # the most recent rolling_window observations available at that time.
        trend_directions: list[str] = []
        for idx in range(len(df)):
            if idx + 1 < self.baseline_n:
                trend_directions.append("Pending")
                continue
            start = max(0, idx - self.rolling_window + 1)
            recent = df.iloc[start : idx + 1]
            _, _, direction = self._slope_and_direction(recent)
            trend_directions.append(direction)
        df["Trend_Direction"] = trend_directions
        return df

    def get_summary(self) -> dict[str, Any] | None:
        df = self.get_dataframe()
        if df.empty:
            return None

        latest = df.iloc[-1]
        measurement_count = len(df)
        baseline_ready = measurement_count >= self.baseline_n
        measurements_to_baseline = max(0, self.baseline_n - measurement_count)

        slope = None
        slope_unit = None
        direction = "Pending"
        direction_ko = "산출 대기"

        if baseline_ready:
            recent = df.tail(self.rolling_window).copy()
            slope, slope_unit, direction = self._slope_and_direction(recent)
            if direction == "Increasing":
                direction_ko = "증가 추세"
            elif direction == "Decreasing":
                direction_ko = "감소 추세"
            elif direction == "Stable":
                direction_ko = "안정적"

        rolling = latest["RollingMean"]
        rolling_value = None if pd.isna(rolling) else float(rolling)
        baseline_value = latest["Personal_Baseline"]
        deviation_value = latest["Deviation_From_Baseline"]

        return {
            "measurement_id": latest["measurement_id"],
            "timestamp": str(latest["measurement_time"]),
            "predicted_class": latest.get("Predicted_Class") or "-",
            "confidence": float(latest.get("Confidence") or 0.0),
            "abnormal_probability": float(latest["Abnormal_Probability"]),
            "baseline": None if pd.isna(baseline_value) else float(baseline_value),
            "deviation": None if pd.isna(deviation_value) else float(deviation_value),
            "rolling_mean": rolling_value,
            "recent_slope": slope,
            "recent_slope_unit": slope_unit,
            "trend_direction": direction,
            "trend_direction_ko": direction_ko,
            "baseline_ready": baseline_ready,
            "measurement_count": measurement_count,
            "measurements_to_baseline": measurements_to_baseline,
        }
