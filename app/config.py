from pathlib import Path
import os

PROJECT_ROOT = Path(os.getenv("RESP_PROJECT_ROOT", r"D:\2026\final_project"))
CHECKPOINT_PATH = Path(
    os.getenv(
        "RESP_CHECKPOINT_PATH",
        str(PROJECT_ROOT / "results" / "trackB" / "checkpoints" / "AlexNet_Base_Final_Selected_seed777.pth"),
    )
)

STORAGE_DIR = Path(os.getenv("RESP_STORAGE_DIR", str(PROJECT_ROOT / "storage")))
DATABASE_PATH = Path(
    os.getenv("RESP_DATABASE_PATH", str(STORAGE_DIR / "respiratory_monitoring.sqlite3"))
)
EXPORT_DIR = Path(os.getenv("RESP_EXPORT_DIR", str(STORAGE_DIR / "exports")))

SAMPLE_RATE = 4000
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 128
FMIN = 0
FMAX = 2000
IMAGE_SIZE = 224

CLASS_NAMES = ["Both", "Crackle", "Normal", "Wheeze"]
BASELINE_N = 3
ROLLING_WINDOW = 3
