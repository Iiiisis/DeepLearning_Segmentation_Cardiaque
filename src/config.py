from pathlib import Path

# ── Données brutes NIfTI ─────────────────────────────────────────────────────
TRAINING_DIR    = Path("Data") / "training"

# ── Données pré-traitées (PNG exportés) ──────────────────────────────────────
DATASET_DIR     = Path("Preprocessed_Dataset")
IMAGE_DIR       = DATASET_DIR / "images"
LABEL_DIR       = DATASET_DIR / "labels"
TESTING_DIR     = DATASET_DIR / "testing"

# ── Sorties du modèle ────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("Output")
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODEL_PATH      = OUTPUT_DIR / "unet_model.h5"

# ── Paramètres d'image ───────────────────────────────────────────────────────
TARGET_SHAPE    = (512, 512)
FRAME_ID        = "frame01"

# ── Hyperparamètres d'entraînement ───────────────────────────────────────────
EPOCHS          = 5
BATCH_SIZE      = 4
THRESHOLD       = 0.5

# ── Création automatique des dossiers ────────────────────────────────────────
for _dir in [IMAGE_DIR, LABEL_DIR, TESTING_DIR, PREDICTIONS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)