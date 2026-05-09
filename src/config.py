"""
config.py
---------
Fichier de configuration global du projet de segmentation cardiaque.
Il définit les chemins vers les données brutes, les dossiers de sortie pour 
les données prétraitées, les chemins de sauvegarde du modèle, ainsi que 
les hyperparamètres d'entraînement et les dimensions des images.
"""

from pathlib import Path

# --- Paramètres d'image ------------------------------------------------------
TARGET_SHAPE    = (512, 512)
FRAME_ID        = "frame01"

# --- Hyperparamètres d'entraînement -----------------------------------------
EPOCHS          = 5         # Nombre de passages complets sur le jeu de données
BATCH_SIZE      = 4         # Nombre d'images traitées avant la mise à jour des poids

# --- Dossier racine pour toutes les sorties générées par le code-------------
OUTPUT_DIR      = Path("Output")

# --- Données brutes (originale) ------------------------------------------------------
TRAINING_DIR    = Path("Data") / "training"
TESTING_DIR     = Path("Data") / "testing"

# --- Données pré-traitées (PNG exportés) ---------------------------------------
DATASET_DIR     = OUTPUT_DIR/ Path("Preprocessed_Dataset")

TRAIN_DIR       = DATASET_DIR/"training_preprocessed"
IMAGE_DIR       = TRAIN_DIR / "images"
LABEL_DIR       = TRAIN_DIR / "labels"

TEST_DIR             = DATASET_DIR/"testing_preprocessed"
IMAGE_TEST_DIR       = TEST_DIR / "images"
LABEL_TEST_DIR       = TEST_DIR / "labels"

# --- Sorties du modèle ------------------------------------------------------
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODEL_PATH      = OUTPUT_DIR / "unet_model.keras"
GRAPH_PATH      = OUTPUT_DIR / "courbes_apprentissage.png"


# --- Création automatique des dossiers ---------------------------------------
for _dir in [IMAGE_DIR, LABEL_DIR, TESTING_DIR, PREDICTIONS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)