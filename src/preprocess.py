import os
import numpy as np
import nibabel as nib
import imageio
import logging
from tqdm import tqdm
from config import DATA_DIR, IMAGE_DIR, LABEL_DIR, IMG_SIZE

# === CONFIGURATION DU LOGGER ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"), # Sauvegarde dans un fichier
        logging.StreamHandler()                  # Affiche aussi dans la console
    ]
)
logger = logging.getLogger(__name__)

def pad_image(image, target_shape=IMG_SIZE):
    """Ajoute du noir autour de l'image pour atteindre la taille cible."""
    padded = np.zeros(target_shape, dtype=image.dtype)
    x, y = image.shape
    x_offset = (target_shape[0] - x) // 2
    y_offset = (target_shape[1] - y) // 2
    padded[x_offset:x_offset+x, y_offset:y_offset+y] = image
    return padded

def run_preprocessing():
    # Création des dossiers
    os.makedirs(str(IMAGE_DIR), exist_ok=True)
    os.makedirs(str(LABEL_DIR), exist_ok=True)
    
    training_path = os.path.join(str(DATA_DIR), "training")
    
    if not os.path.exists(training_path):
        logger.error(f"Le dossier training est introuvable : {training_path}")
        return

    patients = [p for p in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, p))]
    logger.info(f"Début du traitement : {len(patients)} patients trouvés.")

    for patient in tqdm(patients, desc="Extraction"):
        patient_path = os.path.join(training_path, patient)
        
        img_file = os.path.join(patient_path, f"{patient}_frame01.nii")
        lbl_file = os.path.join(patient_path, f"{patient}_frame01_gt.nii")

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
                
                imageio.imwrite(os.path.join(str(IMAGE_DIR), name), img_slice.astype(np.uint8))
                
                label_name = name.replace(".png", "_label.png")
                imageio.imwrite(os.path.join(str(LABEL_DIR), label_name), lbl_slice.astype(np.uint8))
        
        except Exception as e:
            logger.exception(f"Erreur critique lors du traitement du patient {patient}")

if __name__ == "__main__":
    logger.info("Lancement du script de prétraitement")
    run_preprocessing()
    logger.info("Prétraitement effectué.")