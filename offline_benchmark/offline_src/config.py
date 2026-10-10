import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Default root for local-recording discovery. Override with the PHASE_BENCHMARK_DATA_DIR
# environment variable to point at your own recording folder; the default stays inside the
# project so a handed-over copy carries no machine-specific absolute path.
DATA_DIR = Path(os.environ.get("PHASE_BENCHMARK_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PROCESSED_DIR = OUTPUTS_DIR / "processed"

SUBJECTS = [
    subject for subject in range(83)
    if subject not in {39, 68, 69, 78, 79}
]
RECORDING = [1, 2]
ONE_RECORDING_PER_SUBJECT = True
PREFERRED_RECORDING_ORDER = [1, 2]
EEG_CHANNELS = ["Fpz-Cz", "Pz-Oz"]
EPOCH_DURATION = 30.0

SO_PHASE_BAND = (0.5, 1.25)
DELTA_BAND = (0.5, 4.0)
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 12.0)
SIGMA_BAND = (12.0, 15.0)
BETA_BAND = (15.0, 30.0)

TARGET_PHASE = 0.0
PHASE_WINDOW = np.pi / 4
REFRACTORY_SEC = 1.2
AMP_PERCENTILE = 60

AR_PAST_WINDOW_SEC = 6.0
AR_ORDER = 12
AR_PREDICTION_HORIZON_MS = 200.0
AR_HILBERT_PADDING_MS = 500.0
AR_UPDATE_STEP_MS = 50.0
AR_RIDGE_ALPHA = 1e-3
AR_AMP_PERCENTILE = 80
AR_MAX_EPOCHS = 50

FEATURE_CACHE_NAME = "bandpower_features_local_edf.joblib"
LOSO_RANDOM_FOREST_N_ESTIMATORS = 150
LOSO_INCLUDE_GRADIENT_BOOSTING = False
LOSO_MAX_WAKE_TO_DEEP_RATIO = 3.0
LOSO_CHECKPOINT_EVERY_FOLDS = 10
SLEEP_CLASSIFIER_THRESHOLD = 0.87

RANDOM_STATE = 42
