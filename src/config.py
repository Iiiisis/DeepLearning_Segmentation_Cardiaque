from pathlib import Path

# ── Racine du projet (dossier contenant les données brutes) ──────────────────
BASE_DIR = Path(".", "Data")

# ── Données brutes NIfTI ─────────────────────────────────────────────────────
DATA_DIR      = BASE_DIR / "Data" / "Data"
TRAINING_DIR  = DATA_DIR / "training"

# ── Données pré-traitées (PNG exportés) ──────────────────────────────────────
DATASET_DIR   = BASE_DIR / "new_dataset"
IMAGE_DIR     = DATASET_DIR / "images"
LABEL_DIR     = DATASET_DIR / "labels"
TESTING_DIR   = DATASET_DIR / "testing"

# ── Sorties du modèle ────────────────────────────────────────────────────────
PREDICTIONS_DIR = BASE_DIR / "predictions"
MODEL_PATH      = BASE_DIR / "unet_model.h5"

# ── Paramètres d'image ───────────────────────────────────────────────────────
TARGET_SHAPE  = (512, 512)   # taille cible après padding
FRAME_ID      = "frame01"    # identifiant de la coupe temporelle utilisée

# ── Hyperparamètres d'entraînement ───────────────────────────────────────────
EPOCHS        = 5
BATCH_SIZE    = 4
THRESHOLD     = 0.5          # seuil de binarisation des prédictions