from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision import models

from .config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    IMAGE_SIZE,
)


class AlexNetRespiratoryClassifier:
    """Final Track B AlexNet inference service.

    Important: this starter accepts a WAV containing one respiratory cycle.
    Full-recording respiratory-cycle segmentation is intentionally not included here.
    """

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,
        )

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"AlexNet checkpoint not found: {self.checkpoint_path}. "
                "Set RESP_CHECKPOINT_PATH or copy the final checkpoint to the project path."
            )
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

    @staticmethod
    def _fix_length(waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        if n == TARGET_SAMPLES:
            return waveform
        if n < TARGET_SAMPLES:
            total = TARGET_SAMPLES - n
            left = total // 2
            right = total - left
            return F.pad(waveform, (left, right))
        start = (n - TARGET_SAMPLES) // 2
        return waveform[..., start : start + TARGET_SAMPLES]

    def waveform_to_mel(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        waveform = self._to_mono(waveform)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = self._fix_length(waveform)
        mel = self.mel_transform(waveform)
        mel = torch.clamp(mel, min=1e-10)
        mel_db = 10.0 * torch.log10(mel)
        mel_db = mel_db - mel_db.max()
        mel_db = torch.clamp(mel_db, min=-80.0, max=0.0)
        return ((mel_db + 80.0) / 80.0).float()

    @staticmethod
    def prepare_pretrained_input(mel: torch.Tensor) -> torch.Tensor:
        x = mel.unsqueeze(0) if mel.ndim == 3 else mel
        x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        return x.repeat(1, 3, 1, 1)

    def predict_mel(self, mel: torch.Tensor) -> dict:
        x = self.prepare_pretrained_input(mel).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(torch.argmax(probs).item())
        prob_map = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": prob_map,
            "abnormal_probability": float(prob_map["Crackle"] + prob_map["Wheeze"] + prob_map["Both"]),
        }

    def predict_wav_path(self, wav_path: str | Path) -> dict:
        waveform, sr = torchaudio.load(str(wav_path))
        mel = self.waveform_to_mel(waveform, sr)
        result = self.predict_mel(mel)
        result["mel"] = mel
        return result

    def predict_upload(self, file_obj: BinaryIO, filename: str = "upload.wav") -> dict:
        suffix = Path(filename).suffix or ".wav"
        # Windows can prevent torchaudio from reopening an active NamedTemporaryFile.
        # Close the file first, run inference, and then remove it explicitly.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_obj.read())
            temp_path = Path(tmp.name)

        try:
            return self.predict_wav_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
