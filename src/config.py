from pathlib import Path

OUTPUT_DIR      = Path("Output")

# ── Données brutes NIfTI ─────────────────────────────────────────────────────
TRAINING_DIR    = Path("Data") / "training"
TESTING_DIR     = Path("Data") / "testing"

# ── Données pré-traitées (PNG exportés) ──────────────────────────────────────
DATASET_DIR     = OUTPUT_DIR/ Path("Preprocessed_Dataset")
TRAIN_DIR       = DATASET_DIR/"training_preprocessed"
IMAGE_DIR       = TRAIN_DIR / "images"
LABEL_DIR       = TRAIN_DIR / "labels"

TEST_DIR        = DATASET_DIR/"testing_preprocessed"
IMAGE_TEST_DIR       = TEST_DIR / "images"
LABEL_TEST_DIR       = TEST_DIR / "labels"

# ── Sorties du modèle ────────────────────────────────────────────────────────
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODEL_PATH      = OUTPUT_DIR / "unet_model.keras"

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