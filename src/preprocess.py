"""
preprocess.py
-------------
Script de préparation des données. Il lit les volumes 3D au format NIfTI,
extrait chaque tranche (slice) axiale, applique un padding pour atteindre
la taille cible (512x512) et sauvegarde les tranches sous forme d'images PNG.
"""

import os
import imageio
import logging
import warnings
import numpy as np
import nibabel as nib
from tqdm import tqdm
from config import TRAINING_DIR,TESTING_DIR ,TARGET_SHAPE, IMAGE_DIR, LABEL_DIR, TARGET_SHAPE, LABEL_TEST_DIR, IMAGE_TEST_DIR

# === CONFIGURATION DU LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',               # Affiche aussi dans la console
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def pad_image(image, target_shape=TARGET_SHAPE):
    """
    Ajoute du noir (zéros) autour de l'image pour atteindre la taille cible
    sans déformer les proportions de l'IRM originale.
    
    """
    padded = np.zeros(target_shape, dtype=image.dtype)
    x, y = image.shape
    
    # Calcul des marges pour centrer l'image
    x_offset = (target_shape[0] - x) // 2
    y_offset = (target_shape[1] - y) // 2
    
    # Insertion de l'image originale au centre du cadre de zéros
    padded[x_offset:x_offset+x, y_offset:y_offset+y] = image
    return padded

def run_preprocessing(source_dir, output_dir, lbl_target=None):
    """
    Parcourt un dossier de patients, lit les fichiers NIfTI, et exporte 
    les tranches 2D au format PNG.
    """
    # Création des dossiers de destination s'ils n'existent pas
    os.makedirs(str(output_dir), exist_ok=True)
    os.makedirs(str(lbl_target), exist_ok=True)
    
    # Recherche du dossier contenant les patients    
    source_path = os.path.join(str(source_dir))

    if not os.path.exists(source_path):
        logger.error(f"Le dossier source est introuvable : {source_dir}")
        return
    
    # Liste tous les sous-dossiers qui commencent par "patient"
    patients = [p for p in os.listdir(source_path) 
            if os.path.isdir(os.path.join(source_path, p)) and p.startswith("patient")]
    logger.info(f"Début du traitement : {len(patients)} patients trouvés.")

    # Barre de progression pour le suivi du traitement
    for patient in tqdm(patients, desc="Extraction"):
        patient_path = os.path.join(source_path, patient)
        
        # Définition des noms de fichiers attendus pour l'image et sa vérité terrain (label)
        img_file = os.path.join(patient_path, f"{patient}_frame01.nii.gz")
        lbl_file = os.path.join(patient_path, f"{patient}_frame01_gt.nii.gz")

        if not os.path.exists(img_file) or not os.path.exists(lbl_file):
            logger.warning(f"Fichiers manquants pour le patient {patient}. Skip.")
            continue

        try:
            # Chargement des volumes 3D avec nibabel
            img_data = nib.load(img_file).get_fdata()
            lbl_data = nib.load(lbl_file).get_fdata()
            
            # Itération sur l'axe Z (la profondeur) pour extraire chaque tranche 2D
            for z in range(img_data.shape[2]):
                # Redimensionnement (padding) de la tranche
                img_slice = pad_image(img_data[:, :, z])
                # NOUVEAU : On normalise l'IRM pour éviter les "trous noirs"
                if img_slice.max() > 0:
                    img_slice = (img_slice / img_slice.max()) * 255.0
                
                lbl_slice = pad_image(lbl_data[:, :, z])
                
                # Formatage du nom de l'image PNG
                name = f"{patient}_slice{z:03d}.png"
                # Sauvegarde de l'image (convertie en entiers 8 bits)
                imageio.imwrite(os.path.join(str(output_dir), name), img_slice.astype(np.uint8))
                
                # Sauvegarde du masque correspondant
                label_name = name.replace(".png", "_label.png")
                imageio.imwrite(os.path.join(str(lbl_target), label_name), lbl_slice.astype(np.uint8))       
        except Exception as e:
            logger.exception(f"Erreur  lors du traitement de l'image du patient {patient}")

if __name__ == "__main__":
    # 1. Traitement du Training (Images + Labels)
    logger.info("Démarrage du prétraitement Training Data")
    run_preprocessing(TRAINING_DIR, IMAGE_DIR, LABEL_DIR)
    
    # 2. Traitement du Testing (Images uniquement)
    logger.info("Démarrage du prétraitement Testing Data")
    run_preprocessing(TESTING_DIR,IMAGE_TEST_DIR,LABEL_TEST_DIR)
    
    logger.info("Prétraitement terminé.")