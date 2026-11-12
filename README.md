# 개인 맞춤형 호흡음 모니터링 연구용 프로토타입

FastAPI와 PyTorch를 이용해 **호흡주기 단위 WAV**를 분류하고, 개인 기준선과 반복 측정 추이를 시각화하는 연구용 PoC입니다.

> **Research prototype only.** 본 시스템은 의료 진단을 제공하거나 의료진의 임상적 판단을 대체하기 위한 시스템이 아닙니다.

## 현재 구현 상태

- 최종 Track B 모델: **AlexNet Base**
- 입력: 호흡주기 단위 WAV
- 전처리: **4,000 Hz / 4.0 sec / Mel-Spectrogram**
- 분류 클래스: `Normal`, `Crackle`, `Wheeze`, `Both`
- `Abnormal Probability = Crackle + Wheeze + Both`
- 개인 기준선: **초기 3회 측정의 Abnormal Probability 평균**
- 최근 3회 이동평균 및 변화 방향 표시
- 실제 종적 추이 그래프
- Crackle / Wheeze / Both 구성 변화 그래프
- 실제 Mel-Spectrogram 시각화
- AlexNet `features[10]` 기반 Grad-CAM
- FastAPI + Jinja2 단일 대시보드
- 대시보드 내 anchor navigation

## 연구 파이프라인

```text
Cycle WAV
  -> 4 kHz / 4 sec normalization
  -> Mel-Spectrogram
  -> AlexNet Base
  -> Normal / Crackle / Wheeze / Both probabilities
  -> Abnormal Probability
  -> Personal Baseline / Rolling Mean / Trend
  -> FastAPI Dashboard + Grad-CAM
```

## 주요 설정

| 항목 | 값 |
|---|---:|
| Sample rate | 4,000 Hz |
| Fixed duration | 4.0 sec |
| N_FFT | 512 |
| Hop length | 128 |
| N_MELS | 128 |
| Frequency range | 0-2,000 Hz |
| Image size | 224 x 224 |
| Baseline N | 3 |
| Rolling window | 3 |
| Grad-CAM layer | AlexNet `features[10]` |

## 폴더 구조

respiratory-sound-monitoring-source/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ config.py
│  ├─ model_service.py
│  ├─ monitoring.py
│  ├─ gradcam_service.py
│  ├─ storage.py
│  ├─ templates/
│  │  └─ dashboard.html
│  └─ static/
│     ├─ css/
│     │  └─ dashboard.css
│     └─ js/
│        └─ dashboard.js
├─ notebooks/
│  ├─ 02_TrackB_Mel_CNN.ipynb
│  └─ 03_HF_Lung_external_validation.ipynb
├─ .env.example
├─ .gitignore
├─ requirements.txt
└─ README.md

## Git 저장소에 포함하지 않는 파일

다음 항목은 `.gitignore`로 제외합니다.

- ICBHI 원본 WAV
- 전처리된 WAV 및 feature 파일
- HF_Lung 등 외부 원자료
- PyTorch checkpoint (`*.pth`, `*.pt`)
- `results/`, `data_processed/`
- NumPy 배열 및 runtime cache
- 향후 SQLite DB 파일

따라서 **학원 PC와 집 PC 모두 데이터와 checkpoint는 로컬에서 별도로 보관**해야 합니다.

## 모델 checkpoint 준비

기본적으로 다음 경로를 찾습니다.

```text
<project_root>\results\trackB\checkpoints\AlexNet_Base_Final_Selected_seed777.pth
```

예를 들어 저장소를 아래처럼 clone했다면:

```text
D:\2026\final_project\respiratory-sound-monitoring\
```

checkpoint 기본 경로는:

```text
D:\2026\final_project\results\trackB\checkpoints\AlexNet_Base_Final_Selected_seed777.pth
```

입니다.

다른 위치를 사용할 경우 환경변수로 지정할 수 있습니다.

```bat
set RESP_PROJECT_ROOT=D:\2026\final_project
set RESP_CHECKPOINT_PATH=D:\2026\final_project\results\trackB\checkpoints\AlexNet_Base_Final_Selected_seed777.pth
```

`.env.example`에는 예시 경로만 있으며 실제 비밀정보나 모델 파일은 포함하지 않습니다.

## 실행 환경

개발/검증 환경:

- Python 3.11 계열
- PyTorch 2.6.0 + CUDA 12.4 계열
- torchvision 0.21.0
- torchaudio 2.6.0
- FastAPI
- Uvicorn
- Jinja2

이미 `ml_study` conda 환경이 구성되어 있다면 해당 환경을 사용하는 것을 권장합니다.

## 실행

저장소 루트에서:

```bat
conda activate ml_study
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

또는 Windows에서:

```text
run.bat
```

브라우저:

```text
http://127.0.0.1:8000/
```

FastAPI API 문서:

```text
http://127.0.0.1:8000/docs
```

## 중요 제한사항

1. 현재 API는 **한 개의 respiratory-cycle WAV**를 입력으로 사용합니다.
2. 긴 원본 녹음에서 respiratory cycle을 자동 분할하는 기능은 아직 포함하지 않았습니다.
3. ICBHI는 실제 개인 종적 반복 측정 데이터셋이 아니므로, baseline/trend 기능은 기술적 PoC입니다.
4. `Abnormal Probability`는 모델 출력 확률의 합이며 **질병 위험도나 임상적 위험도**를 의미하지 않습니다.
5. Grad-CAM은 분류 결과에 상대적으로 기여한 영역을 보여주는 설명가능성 도구이며 확정적인 임상적 근거가 아닙니다.

## 다음 개발 단계

- SQLite/CSV 기반 측정 결과 영구 저장
- 결과 내보내기 및 대상자별 기록 관리
- 측정 이력에서 특정 측정의 Mel/Grad-CAM 재조회
- 배치 평가 화면
- 코호트 통계 화면

## 현재 안정화 버전

GitHub 업로드 기준: **FastAPI dashboard v11 stable prototype**

- 연속 4회 이상 추론 검증
- 초기 3회 baseline 형성 검증
- 종적 추이 및 component trend 실제 데이터 연동
- 실제 Mel-Spectrogram / Grad-CAM 연동
- 대시보드 anchor navigation 적용
