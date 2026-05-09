import os
import numpy as np
import nibabel as nib
import imageio
import logging
from tqdm import tqdm
from config import TRAINING_DIR,TESTING_DIR ,TARGET_SHAPE, IMAGE_DIR, LABEL_DIR, TARGET_SHAPE, LABEL_TEST_DIR, IMAGE_TEST_DIR

# === CONFIGURATION DU LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',               # Affiche aussi dans la console
)
logger = logging.getLogger(__name__)

def pad_image(image, target_shape=TARGET_SHAPE):
    """Ajoute du noir autour de l'image pour atteindre la taille cible"""
    padded = np.zeros(target_shape, dtype=image.dtype)
    x, y = image.shape
    x_offset = (target_shape[0] - x) // 2
    y_offset = (target_shape[1] - y) // 2
    padded[x_offset:x_offset+x, y_offset:y_offset+y] = image
    return padded

def run_preprocessing(source_dir, output_dir, lbl_target=None):
    # Création des dossiers
    os.makedirs(str(output_dir), exist_ok=True)
    os.makedirs(str(lbl_target), exist_ok=True)
    
    #Recherche des patients
    source_path = os.path.join(str(source_dir))

    if not os.path.exists(source_path):
        logger.error(f"Le dossier source est introuvable : {source_dir}")
        return
    
    # Parcourt des patients
    patients = [p for p in os.listdir(source_path) 
            if os.path.isdir(os.path.join(source_path, p)) and p.startswith("patient")]
    
    logger.info(f"Début du traitement : {len(patients)} patients trouvés.")

    for patient in tqdm(patients, desc="Extraction"):
        patient_path = os.path.join(source_path, patient)
        
        img_file = os.path.join(patient_path, f"{patient}_frame01.nii.gz")
        lbl_file = os.path.join(patient_path, f"{patient}_frame01_gt.nii.gz")

        if not os.path.exists(img_file) or not os.path.exists(lbl_file):
            logger.warning(f"Fichiers manquants pour le patient {patient}. Skip.")
            continue

        try:
            img_data = nib.load(img_file).get_fdata()
            lbl_data = nib.load(lbl_file).get_fdata()

            for z in range(img_data.shape[2]):
                img_slice = pad_image(img_data[:, :, z])
                lbl_slice = pad_image(lbl_data[:, :, z])
                
                name = f"{patient}_slice{z:03d}.png"
                
                imageio.imwrite(os.path.join(str(output_dir), name), img_slice.astype(np.uint8))
                
                label_name = name.replace(".png", "_label.png")
                imageio.imwrite(os.path.join(str(LABEL_TEST_DIR), label_name), lbl_slice.astype(np.uint8))       
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